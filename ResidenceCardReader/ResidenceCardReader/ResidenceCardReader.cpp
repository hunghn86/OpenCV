// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "baseapi.h"
#include <opencv2/opencv.hpp>
#include "allheaders.h"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace tesseract;

const char* keys =
{
	"{help h||}{@image |../data/fruits.jpg|input image name}"
};

/// Global variables
const char* source_window = "Source image";
const char* finalImg = "finalImg.png";
const char* binaryImg = "binaryImg.jpg";
int thresh = 0;
int max_thresh = 60;
Mat src_image; Mat src_gray; Mat blur_image; Mat input_img;

struct userdata {
	Mat im;
	vector<Point2f> points;
};

struct str {
	bool operator() (Point2f a, Point2f b) {
		if (a.y != b.y)
			return a.y < b.y;
		return a.x <= b.x;
	}
} comp;

static void help()
{
	printf("\nThis sample demonstrates Canny edge detection\n"
		"Call:\n"
		"    /.edge [image_name -- Default is ../data/fruits.jpg]\n\n");
}

bool isBadContours(vector<Point> contours) {
	bool isBadContours = false;
	Rect boundRect = boundingRect(contours);
	double area = boundRect.width*boundRect.height;
	double ratio = boundRect.height / boundRect.width;
	if (area < 21 || area > 5000) { // || ratio < 1 || ratio > 2) {
		isBadContours = true;
	}
	return isBadContours;
}

void getTextByTessAPI() {
	char *outText;
	//TCHAR Buffer[255];
	//DWORD dwRet;
	//dwRet = GetCurrentDirectory(255, Buffer);
	//FILE * pFile;
	//pFile = fopen("../lib/tessdata/jpn.traineddata", "r");
	//if (pFile != NULL)
	//{
	//	fclose(pFile);
	//}

	TessBaseAPI *api = new TessBaseAPI();

	if (api->Init("../lib/", "jpn")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		return;
	}

	// Open input image with leptonica library
	Pix *image = pixRead(binaryImg);
	api->SetImage(image);
	// Get OCR result
	outText = api->GetUTF8Text();

	ofstream myfile;
	myfile.open("result.txt");
	myfile << outText;
	myfile.close();

	// Destroy used object and release memory
	api->End();
	//delete[] outText;
	pixDestroy(&image);
	return;
}

/** @function getFinalImage */
void getFinalImage(int, void*)
{
	Mat canny_output, binaryMat;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Scalar color = Scalar(255, 255, 255);

	// Converts image to gray
	cvtColor(input_img, src_gray, CV_BGR2GRAY);

	// Remove noise using blur
	blur(src_gray, blur_image, Size(3, 3));

	// Detect edges using canny
	Canny(blur_image, canny_output, thresh, thresh * 3, 3, true);

	// Find contours
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	Mat result = Mat::zeros(input_img.size(), CV_8UC3);
	vector<Rect> boundRect(contours.size());
	threshold(result, result, 255, 255, THRESH_BINARY_INV);

	for (int i = 0; i < contours.size(); i++)
	{
		if (isBadContours(contours[i])) {
			continue;
		}
		boundRect[i] = boundingRect(contours[i]);
		for (int j = boundRect[i].y; j < boundRect[i].y + boundRect[i].height; j++)
			for (int k = boundRect[i].x; k < boundRect[i].x + boundRect[i].width; k++)
			{
				result.at<Vec3b>(j, k) = input_img.at<Vec3b>(j, k);
			}
	}

	namedWindow("final", CV_WINDOW_AUTOSIZE);
	imshow("final", result);
	cvtColor(result, src_gray, CV_BGR2GRAY);
	threshold(src_gray, binaryMat, 130, 255, cv::THRESH_BINARY);
	imshow("binaryMat", binaryMat);
	imwrite("binaryImg.jpg", binaryMat);
	// Write tmp image to recognise
	imwrite(finalImg, result);

	// Recognise using tesseract
	getTextByTessAPI();
}

void rotatePerspective(Mat im_src, vector<Point2f> pts_src)
{
	// Destination image. The aspect ratio of the book is 3/4
	Size size(450, 300);
	Mat im_dst = Mat::zeros(size, CV_8UC3);

	// Create a vector of destination points.
	vector<Point2f> pts_dst;
	pts_dst.push_back(Point2f(0, 0));
	pts_dst.push_back(Point2f(size.width - 1, 0));
	pts_dst.push_back(Point2f(size.width - 1, size.height - 1));
	pts_dst.push_back(Point2f(0, size.height - 1));

	// Calculate the homography
	Mat h = findHomography(pts_src, pts_dst);

	// Warp source image to destination
	warpPerspective(im_src, im_dst, h, size);
	imshow("warpPerspective", im_dst);

	// Get final image
	input_img = im_dst.clone();
	createTrackbar(" getFinalImage:", "rotatedImage", &thresh, max_thresh, getFinalImage);
	getFinalImage(0, 0);
}

vector<Point2f> order_points(vector<Point> contours) {
	vector<Point2f> ret_pts;
	int minPos = 0, maxPos = 0;
	int minSumPos = 0, maxSumPos = 0;
	const int kSize = contours.size();
	int *sum = new int[kSize];
	int *diff = new int[kSize];

	for (int i = 0; i < contours.size(); i++) {
		diff[i] = contours[i].y - contours[i].x;
		sum[i] = contours[i].y + contours[i].x;
	}

	int maxDiff = diff[0], minDiff = diff[0];
	for (int j = 1; j < contours.size(); j++) {
		if (maxDiff < diff[j]) {
			maxDiff = diff[j];
			maxPos = j;
		}
		if (minDiff > diff[j]) {
			minDiff = diff[j];
			minPos = j;
		}
	}

	int maxSum = sum[0], minSum = sum[0];
	for (int j = 1; j < contours.size(); j++) {
		if (maxSum < sum[j]) {
			maxSum = sum[j];
			maxSumPos = j;
		}
		if (minSum > sum[j]) {
			minSum = sum[j];
			minSumPos = j;
		}
	}

	ret_pts.push_back(contours[minSumPos]);
	ret_pts.push_back(contours[minPos]);
	ret_pts.push_back(contours[maxSumPos]);
	ret_pts.push_back(contours[maxPos]);

	delete[] sum;
	delete[] diff;
	return ret_pts;
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Scalar color = Scalar(0, 255, 0);

	/// Detect edges using canny
	blur(src_gray, blur_image, Size(3, 3));
	Canny(blur_image, canny_output, thresh, thresh * 3, 3);

	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	/// Merge contours
	vector<Point> merged_contour_points;
	for (int i = 0; i < contours.size(); i++)
	{
		for (int j = 0; j < contours[i].size(); j++) {
			merged_contour_points.push_back(contours[i][j]);
		}
	}

	vector<Point2f> pts_src = order_points(merged_contour_points);
	rotatePerspective(src_image, pts_src);
}

/** @function findContourInImage **/
void findContourInImage(Mat src)
{
	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	/// Create Window
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

	createTrackbar(" Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);
}

int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];

	// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.lpstrFile = szFile;

	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	if (GetOpenFileName(&ofn) == TRUE) {
		/// Load source image and convert it to gray
		src_image = imread(ofn.lpstrFile, IMREAD_COLOR);
		findContourInImage(src_image);
	}

	waitKey(0);
	return 0;
}