using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV;
using Microsoft.AspNetCore.Cors;
using Microsoft.AspNetCore.Mvc;
using System.Drawing;

namespace NespressoCupsDetection;


[ApiController]
[Route("[controller]")]
[EnableCors("AllowAllOrigins")]
public class ImageRecognitionController : Controller
{
    public static readonly string _imagesFolder = Path.Combine(Environment.CurrentDirectory, "images");
    public static readonly string _debugFolder = Path.Combine(_imagesFolder, "debug");

    public static readonly string _samplesFolder = Path.Combine(_imagesFolder, "samples");

    static readonly string[] templatePaths = Directory.GetFiles(_samplesFolder, "*.jpg");
    public static readonly List<string> expectedOutcomes = new List<string> { "VOLTESSO", "ORAFIO", "BIANCO", "PICOLLO", "DOLCE" };

    [HttpPost("ProcessImage")]
    public async Task<IActionResult> ProcessImage1([FromBody] ImageUploadModel model)
    {
        if (model?.Image == null)
            return BadRequest("No image received");
        try
        {

            string resultedText = string.Empty;
            var imageBytes = Convert.FromBase64String(model.Image.Replace("data:image/png;base64,", "").Replace("data:image/jpeg;base64,", ""));
            Mat inputImage = new Mat();
            CvInvoke.Imdecode(imageBytes, ImreadModes.Unchanged, inputImage);
            inputImage.Save(Path.Combine(_debugFolder, "imagineNoua.jpg"));
            Mat inputGray = new Mat();
            CvInvoke.CvtColor(inputImage, inputGray, ColorConversion.Bgr2Gray);
            inputGray.ConvertTo(inputGray, DepthType.Cv32F);
            inputImage.Save(Path.Combine(_debugFolder, "imagineGri.jpg"));

            var validScales = new List<double> { 1.75, 2 };
            Mat exclusionMask = CreateExclusionMask(inputGray);

            foreach (var templatePath in templatePaths)
            {

                Mat templateGray = new Mat();
                Mat template = CvInvoke.Imread(templatePath, ImreadModes.Unchanged);
                CvInvoke.CvtColor(template, templateGray, ColorConversion.Bgr2Gray);
                templateGray.ConvertTo(templateGray, DepthType.Cv32F);

                foreach (var scale in validScales)
                {
                    Mat resizedTemplate = ResizeTemplate(templateGray, scale);
                    // Check dimensions before applying MatchTemplate
                    if (resizedTemplate.Width > inputGray.Width || resizedTemplate.Height > inputGray.Height)
                    {
                        Console.WriteLine($"Template prea mare: {resizedTemplate.Width}x{resizedTemplate.Height} vs {inputGray.Width}x{inputGray.Height}");
                        continue;
                    }
                    Mat result = new Mat();
                    CvInvoke.MatchTemplate(inputGray, resizedTemplate, result, Emgu.CV.CvEnum.TemplateMatchingType.CcoeffNormed);
                    float[,] resultData = result.GetData() as float[,];

                    for (int y = 0; y < result.Rows; y++)
                    {
                        for (int x = 0; x < result.Cols; x++)
                        {
                            if (resultData[y, x] > 0.55)
                            {
                                Point matchPoint = new Point(x - 20, y - 30);
                                Rectangle matchRect = new Rectangle(matchPoint, new Size(resizedTemplate.Width + 40, resizedTemplate.Height + 60));
                                // Salvează subimaginea în debug folder cu coordonatele găsite
                                Mat subImage = new Mat(inputImage, matchRect);
                                string subImageFilePath = Path.Combine(_debugFolder, $"match_{matchPoint.X}_{matchPoint.Y}_{matchRect.Width}x{matchRect.Height}.jpg");
                                subImage.Save(subImageFilePath);

                                if (IsRegionFree(exclusionMask, matchRect, 70))
                                {
                                    ExcludeRegionFromMask(exclusionMask, matchRect);

                                    CvInvoke.Rectangle(inputImage, matchRect, new MCvScalar(0, 255, 0), 2);

                                    var detectedText = DetectTextWithOCR(subImage);
                                    if (!string.IsNullOrEmpty(detectedText))
                                    {
                                        resultedText = string.Concat(resultedText, " , ", detectedText);
                                        //Point textPosition = new Point(matchRect.X, matchRect.Y + 15);
                                        //CvInvoke.PutText(inputImage, detectedText, textPosition, FontFace.HersheyPlain, 1.2, new MCvScalar(50, 120, 255), 3);
                                    }


                                }
                            }

                        }
                    }
                }
            }
            // Convertim Mat la Base64
            VectorOfByte buffer = new VectorOfByte();
            CvInvoke.Imencode(".png", inputImage, buffer);
            if (string.IsNullOrEmpty(resultedText))
                resultedText = "Nu s-a putut identifica";
            //string base64Image = Convert.ToBase64String(buffer.ToArray());
            //return Ok(new { processedImage = "data:image/png;base64," + base64Image });
            return Ok(new { resulttext = resultedText });

        }
        catch (Exception ex)
        {
            return BadRequest(ex.Message);
        }
    }

    public static List<Mat> IdentifyNespressoBoxes(Mat image)
    {
        List<Mat> boxFaces = new List<Mat>();

        // 1. Preprocesare imagine
        Mat grayImage = new Mat();
        CvInvoke.CvtColor(image, grayImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
        CvInvoke.GaussianBlur(grayImage, grayImage, new Size(3, 3), 0);
        CvInvoke.EqualizeHist(grayImage, grayImage);

        // 2. Detectarea contururilor
        Mat cannyEdges = new Mat();
        CvInvoke.Canny(grayImage, cannyEdges, 50, 150); // Parametrii Canny pot fi ajustați
        cannyEdges.Save(Path.Combine(_debugFolder, "canny.jpg"));

        // 3. Găsirea contururilor dreptunghiulare
        List<RotatedRect> rectangles = new List<RotatedRect>();
        using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
        {
            CvInvoke.FindContours(cannyEdges, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

            double totalArea = 0;
            for (int i = 0; i < contours.Size; i++)
            {
                using (VectorOfPoint contour = contours[i])
                {
                    totalArea += CvInvoke.ContourArea(contour);
                }
            }
            double averageArea = contours.Size > 0 ? totalArea / contours.Size : 0;

            for (int i = 0; i < contours.Size; i++)
            {
                using (VectorOfPoint contour = contours[i])
                {
                    double area = CvInvoke.ContourArea(contour);

                    // Ajustăm aria minimă dinamic, bazat pe dimensiunea imaginii și aria medie
                    double minArea = Math.Max(5000, averageArea * 0.1); // 10% din aria medie, dar minim 5000

                    if (area > minArea)
                    {
                        RotatedRect box = CvInvoke.MinAreaRect(contour);

                        // Verificăm și numărul de colțuri, pe lângă raportul aspectului
                        if (IsRectangle(box, contour))
                        {
                            rectangles.Add(box);
                        }
                    }
                }
            }
        }

        foreach (RotatedRect rectangle in rectangles)
        {
            PointF[] points = new PointF[4];
            points[0] = new PointF(rectangle.Center.X + rectangle.Size.Width / 2, rectangle.Center.Y + rectangle.Size.Height / 2);
            points[1] = new PointF(rectangle.Center.X - rectangle.Size.Width / 2, rectangle.Center.Y + rectangle.Size.Height / 2);
            points[2] = new PointF(rectangle.Center.X - rectangle.Size.Width / 2, rectangle.Center.Y - rectangle.Size.Height / 2);
            points[3] = new PointF(rectangle.Center.X + rectangle.Size.Width / 2, rectangle.Center.Y - rectangle.Size.Height / 2);

            Point[] intPoints = new Point[points.Length];
            for (int i = 0; i < points.Length; i++)
            {
                intPoints[i] = new Point((int)points[i].X, (int)points[i].Y);
            }

            Mat mask = new Mat(image.Size, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(0));

            using (VectorOfPoint vp = new VectorOfPoint())
            {
                vp.Push(intPoints); // Adăugăm punctele la VectorOfPoint
                CvInvoke.FillConvexPoly(mask, vp, new MCvScalar(255));
            }


            Mat boxFace = new Mat();
            image.CopyTo(boxFace, mask);

            boxFaces.Add(boxFace);
        }

        return boxFaces;
    }
    private static bool IsRectangle(RotatedRect box, VectorOfPoint contour)
    {
        float aspectRatio = box.Size.Width / box.Size.Height;
        if (!(aspectRatio > 0.5 && aspectRatio < 2)) return false;

        // Verificăm și numărul de colțuri
        double perimeter = CvInvoke.ArcLength(contour, true);
        using (VectorOfPoint approxContour = new VectorOfPoint())
        {
            CvInvoke.ApproxPolyDP(contour, approxContour, 0.02 * perimeter, true); // Ajustați epsilonul
            int numberOfCorners = approxContour.Size;
            return numberOfCorners >= 4 && numberOfCorners <= 6; // Acceptăm între 4 și 6 colțuri
        }
    }

    public static string DetectTextWithOCR(Mat image, string? imagePath = null)
    {
        if (image == null)
            image = CvInvoke.Imread(imagePath, ImreadModes.Unchanged);

        string PerformOCR(Mat img)
        {
            using (Tesseract tesseract = new Tesseract("D:\\Practice\\teste\\tessdata", "eng", OcrEngineMode.TesseractLstmCombined))
            {
                //piedone-dupa teste am observat ca cele mai bune rezultate le a obtinut PageSegMode.Auto(le am testat pe toate aproape)
                tesseract.PageSegMode = PageSegMode.Auto;
                tesseract.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz  ");
                tesseract.SetVariable("classify_enable_learning", "0");
                tesseract.SetImage(img);
                tesseract.Recognize();
                return tesseract.GetUTF8Text().Trim();
            }
        }

        Mat grayImage = new Mat();
        CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
        //piedone-filtrul gausian ajuta semnificativ, testat cu valori de la Size(1, 1) la Size(3, 3)
        //CvInvoke.GaussianBlur(grayImage, grayImage, new Size(1, 1), 0);
        CvInvoke.CLAHE(grayImage, 2.0, new Size(8, 8), grayImage);

        CvInvoke.Threshold(grayImage, grayImage, 0, 255, ThresholdType.Otsu | ThresholdType.Binary);
        Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(1, 1), new Point(-1, -1));
        //piedone-dupa teste am observat ca cele mai bune rezultate le a obtinut MorphOp.Dilate
        CvInvoke.MorphologyEx(grayImage, grayImage, MorphOp.Erode, kernel, new Point(-1, -1), 5, BorderType.Default, new MCvScalar());
        Random nx = new Random();

        grayImage.Save(Path.Combine(_debugFolder, $"debugOCR{nx.Next()}.jpg"));
        string label = PerformOCR(grayImage);
        if (!string.IsNullOrEmpty(label) && expectedOutcomes.Any(expected => IsSimilar(label, expected)))
        {
            var res = expectedOutcomes.Where(expected => IsSimilar(label, expected)).Select(x => x).First();
            return res;
        }

        Mat[] rotatedImages = RotateImageMultipleTimes(grayImage);
        foreach (Mat rotatedImage in rotatedImages)
        {
            label = PerformOCR(rotatedImage);
            if (!string.IsNullOrEmpty(label) && expectedOutcomes.Any(expected => IsSimilar(label, expected)))
            {
                var res = expectedOutcomes.Where(expected => IsSimilar(label, expected)).Select(x => x).First();
                return res;
            }
        }

        return string.Empty;
    }

    [HttpPost("DetectShapes")]
    public async Task<IActionResult> DetectShapes([FromBody] ImageUploadModel model, bool returnImage = false)
    {
        try
        {
            var imageBytes = Convert.FromBase64String(model.Image.Replace("data:image/png;base64,", "").Replace("data:image/jpeg;base64,", ""));
            Mat inputImage = new Mat();
            CvInvoke.Imdecode(imageBytes, ImreadModes.Unchanged, inputImage);
            inputImage.Save(Path.Combine(_debugFolder, "imagineInitiala.jpg"));
            //Mat processed = new Mat();
            var shapesfound = FindColoredSquares(inputImage, out Mat processed);
            string resultedText = string.Empty;
            Random rnd = new Random();

            foreach (var shape in shapesfound)
            {
                shape.Save(Path.Combine(_debugFolder, $"debug{rnd.Next()}.jpg"));

                var detectedText = DetectTextWithOCR(shape);
                if (!string.IsNullOrEmpty(detectedText))
                    resultedText = string.Concat(resultedText, " , ", detectedText);

            }
            var result = resultedText
                   .Split(',')
                   .Select(p => p.Trim())
                   .Where(p => !string.IsNullOrEmpty(p))
                   .GroupBy(p => p)
                   .Select(g => $"{g.Count()} x {g.Key}")
                   .Aggregate((a, b) => $"{a} \n {b}");
            if (returnImage)
            {
                byte[] processedImageBytes = CvInvoke.Imencode(".jpg", processed);
                string processedImageBase64 = Convert.ToBase64String(processedImageBytes);
                return Ok(new { resulttext = result, image = $"data:image/jpeg;base64,{processedImageBase64}" });
            }
            return Ok(new { resulttext = result, image = "" });

        }
        catch (Exception ex) { return BadRequest(ex.Message); }
    }


    //public static Mat[] FindColoredSquares(Mat inputImage)
    //{
    //    Mat blurred = new Mat();
    //    CvInvoke.GaussianBlur(inputImage, blurred, new Size(1, 1), 1);

    //    Mat hsv = new Mat();
    //    CvInvoke.CvtColor(blurred, hsv, ColorConversion.Bgr2Hsv);

    //    Mat binary = new Mat();
    //    binary.Save(Path.Combine(_debugFolder, "afterProcessing.jpg"));
    //    VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
    //    Mat hierarchy = new Mat();
    //    CvInvoke.FindContours(binary, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxSimple);

    //    List<Mat> foundSquares = new List<Mat>();

    //    foreach (var contour in contours.ToArrayOfArray())
    //    {
    //        double peri = CvInvoke.ArcLength(new VectorOfPoint(contour), true);
    //        VectorOfPoint approx = new VectorOfPoint();
    //        CvInvoke.ApproxPolyDP(new VectorOfPoint(contour), approx, 0.04 * peri, true);

    //        Rectangle rect = CvInvoke.BoundingRectangle(approx);
    //        float aspectRatio = (float)rect.Width / rect.Height;

    //        double area = rect.Width * rect.Height;

    //        if (rect.Width > 300 && rect.Height > 300 && rect.Width < 1000 && rect.Height < 1000 && aspectRatio > 0.7 && aspectRatio < 1.3) // Criterii mai stricte
    //        {
    //            Mat subImage = new Mat(inputImage, rect);
    //            foundSquares.Add(subImage);
    //        }
    //    }
    //    return foundSquares.ToArray();
    //}
    public static Mat ConvertToAppropriateColorSpace(Mat inputImage)
    {
        // 1. Conversia la HSV
        Mat hsvImage = new Mat();
        CvInvoke.CvtColor(inputImage, hsvImage, ColorConversion.Bgr2Hsv);

        // 2. Extragerea canalului de culoare relevant (exemplu: canalul H)
        // Puteți experimenta cu canalele H, S sau V pentru a vedea care funcționează cel mai bine
        Mat colorChannel = new Mat();
        CvInvoke.ExtractChannel(hsvImage, colorChannel, 2); // 0 pentru H, 1 pentru S, 2 pentru V

        // 3. (Opțional) Egalizarea histogrammei canalului de culoare pentru a îmbunătăți contrastul
        Mat equalizedChannel = new Mat();
        CvInvoke.EqualizeHist(colorChannel, equalizedChannel);

        // 4. (Opțional) Revenirea la spațiul de culoare original (BGR) cu canalul de culoare modificat
        Mat resultImage = new Mat();
        CvInvoke.InsertChannel(equalizedChannel, hsvImage, 0); // Inserarea canalului modificat în imaginea HSV
        CvInvoke.CvtColor(hsvImage, resultImage, ColorConversion.Hsv2Bgr);

        return resultImage;
    }

    public static Mat[] FindColoredSquares(Mat inputImage, out Mat processedImage)
    {
        //// Convertire la gri
        //Mat gray = new Mat();
        //CvInvoke.CvtColor(inputImage, gray, ColorConversion.Bgr2Gray);
        Mat hsvChannel = ConvertToAppropriateColorSpace(inputImage);
        //Mat blurred = new Mat();
        //CvInvoke.GaussianBlur(hsvChannel, blurred, new Size(5, 5), 10);
        //CvInvoke.MedianBlur(blurred, blurred, 7);
        Mat blurred = new Mat();
        CvInvoke.BilateralFilter(hsvChannel, blurred, 15, 150, 300); // Parametri ajustabili

        // Detectare margini Canny
        Mat edges = new Mat();
        CvInvoke.Canny(blurred, edges, 100, 300);

        // Operație morfologică pentru a îmbunătăți contururile
        Mat morph = new Mat();
        Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2, 2), new Point(-1, -1));
        //CvInvoke.MorphologyEx(edges, morph, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Replicate, new MCvScalar(255, 255, 255));
        CvInvoke.MorphologyEx(edges, morph, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Replicate, new MCvScalar(255, 255, 255));

        // Detectare contururi
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        Mat hierarchy = new Mat();

        morph.Save(Path.Combine(_debugFolder, "afterProcessing.jpg"));

        CvInvoke.FindContours(morph, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxTc89Kcos);

        List<Mat> foundRectangles = new List<Mat>();


        foreach (var contour in contours.ToArrayOfArray())
        {
            double peri = CvInvoke.ArcLength(new VectorOfPoint(contour), true);
            VectorOfPoint approx = new VectorOfPoint();
            CvInvoke.ApproxPolyDP(new VectorOfPoint(contour), approx, 0.04 * peri, true);

            Rectangle rect = CvInvoke.BoundingRectangle(approx);
            float aspectRatio = (float)rect.Width / rect.Height;

            //double area = rect.Width * rect.Height;
            //&& rect.Width < 1000 && rect.Height < 1000 && aspectRatio > 0.7 && aspectRatio < 1.3
            if (rect.Width > 300 && rect.Height > 300 && rect.Width < 1000 && rect.Height < 1000)
            {
                Mat subImage = new Mat(inputImage, rect);
                foundRectangles.Add(subImage);

                CvInvoke.Rectangle(inputImage, rect, new MCvScalar(0, 255, 0), 6); // Desenăm dreptunghiul
            }
        }

        inputImage.Save("DetectedRectangles.jpg");
        processedImage = inputImage.Clone();

        return foundRectangles.ToArray();
    }


    public static Mat[] RotateImageMultipleTimes(Mat subImage, int rotations = 15, double angleStep = 15)
    {
        Mat[] rotatedImages = new Mat[rotations];
        Size size = subImage.Size;
        PointF center = new PointF(size.Width / 2f, size.Height / 2f);

        for (int i = 0; i < rotations; i++)
        {
            double angle = angleStep * (i + 1);
            Mat rotationMatrix = new Mat();
            CvInvoke.GetRotationMatrix2D(center, angle, 1.0, rotationMatrix);
            Mat rotated = new Mat();
            CvInvoke.WarpAffine(subImage, rotated, rotationMatrix, size, Inter.Linear, Warp.Default, BorderType.Constant, new MCvScalar(0, 0, 0));
            rotatedImages[i] = rotated;
        }

        return rotatedImages;
    }


    public static Mat CreateExclusionMask(Mat image)
    {
        Mat exclusionMask = new Mat(image.Size, DepthType.Cv8U, 1);
        exclusionMask.SetTo(new MCvScalar(0));
        return exclusionMask;
    }

    public static Mat ResizeTemplate(Mat template, double scale)
    {
        Mat resizedTemplate = new Mat();
        CvInvoke.Resize(template, resizedTemplate, new Size(0, 0), scale, scale, Emgu.CV.CvEnum.Inter.Linear);
        return resizedTemplate;
    }

    public static bool IsRegionFree(Mat mask, Rectangle region, int overlapTolerance = 30)
    {
        if (region.X < 0 || region.Y < 0 ||
            region.X + region.Width > mask.Cols ||
            region.Y + region.Height > mask.Rows ||
            region.Width <= 0 || region.Height <= 0)
        {
            return false; // Region invalid
        }

        Mat roi = new Mat(mask, region);
        MCvScalar mean = CvInvoke.Mean(roi);

        // Calculăm procentajul pixelilor liberi (valoare 0)
        double freeRatio = mean.V0 / 255.0;

        // Permitem overlap dacă cel puțin 30% din regiune este liberă
        return freeRatio < (overlapTolerance / 100.0);
    }

    public static void ExcludeRegionFromMask(Mat mask, Rectangle region)
    {
        CvInvoke.Rectangle(mask, region, new MCvScalar(255), -1); // Mark the region with white (255) to indicate exclusion
    }

    public static bool IsSimilar(string label, string expected, int maxDistance = 2)
    {
        int distance = LevenshteinDistance(label, expected);
        return distance <= maxDistance;
    }
    public static int LevenshteinDistance(string s1, string s2)
    {
        int[,] dp = new int[s1.Length + 1, s2.Length + 1];

        for (int i = 0; i <= s1.Length; i++)
            dp[i, 0] = i;
        for (int j = 0; j <= s2.Length; j++)
            dp[0, j] = j;

        for (int i = 1; i <= s1.Length; i++)
        {
            for (int j = 1; j <= s2.Length; j++)
            {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                dp[i, j] = Math.Min(Math.Min(
                    dp[i - 1, j] + 1,      // Ștergere
                    dp[i, j - 1] + 1),     // Inserare
                    dp[i - 1, j - 1] + cost); // Înlocuire
            }
        }

        return dp[s1.Length, s2.Length];
    }

}


public class ImageUploadModel
{
    public string Image { get; set; }
}