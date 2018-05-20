// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <limits>
#include "math.h"

using namespace std;

//Segment the image with grabcut
void segmentImage() {

	int cnt = 1;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);

		bool showCrosshair = false;
		bool fromCenter = false;
		Rect rectangle = selectROI("Image", src, fromCenter, showCrosshair);
		
		Mat result; // segmentation result (4 possible values)
		Mat bgModel, fgModel; // the models (internally used)

		// GrabCut segmentation
		grabCut(src,// input image
			result, // segmentation result
			rectangle,// rectangle containing foreground
			bgModel, fgModel, // models
			1,        // number of iterations
			cv::GC_INIT_WITH_RECT); // use rectangle
	
		// Get the pixels marked as likely foreground
		compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
		// Generate output image
		Mat foreground(src.size(), CV_8UC3);
		//cv::Mat background(image.size(),CV_8UC3,cv::Scalar(255,255,255));
		src.copyTo(foreground, result); // bg pixels not copied

		// draw rectangle on original image
		cv::rectangle(src, rectangle, cv::Scalar(255, 255, 255), 1);
		std::string file_name = "ExtractedImgs/Foreground";
		file_name += std::to_string(cnt++);
		file_name += ".jpg";
		std::cout << file_name;
		imwrite(file_name, foreground);
		imshow(file_name, foreground);
		Mat background;
		src.copyTo(background,~result);
		imshow("Background.jpg", background);

		imshow("image", src);
		waitKey();
	}
}

vector<vector<int>> calculateHistogram(Mat img)
{	
	vector<int> red(256);
	vector<int> green(256);
	vector<int> blue(256);
	vector<vector<int>> histogram(3);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			green[img.at<Vec3b>(i, j)[0]]++;
			blue[img.at<Vec3b>(i, j)[1]]++;
			red[img.at<Vec3b>(i, j)[2]]++;
		}
	}
	histogram[0] = green;
	histogram[1] = blue;
	histogram[2] = red;
	return histogram;

}
/*compare the histograms of the two images, h1 - image under test, h2 - already classified image*/
float compareHistograms(vector<vector<int>> h1, vector<vector<int>> h2) {

	float sum = 0;
	for (int i = 0; i < 256; i++) {
		if (h1[0][i] != 0) {
			sum += pow(h1[0][i] - h2[0][i], 2) / h1[0][i];
		}
		if (h1[1][i] != 0) {
			sum += pow(h1[1][i] - h2[1][i], 2) / h1[1][i];
		}
		if (h1[2][i] != 0) {
			sum += pow(h1[2][i] - h2[2][i], 2) / h1[2][i];
		}
	}
	return sum / 3;
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, vector<int> hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	int max_hist = 0;
	int prev = 0;
	int curr = 0;
	for (int i = 0; i<hist_cols; i++)
		if (hist[i] > max_hist) {
			max_hist = hist[i];
			prev = curr;
			curr = i;
		}
			
	//eliminate the maximum value -- most likely will be a background pixel value
	hist[curr] = 0;

	//set the heigh of the histogram to be the second biggest bin
	max_hist = hist[prev];

	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void calculateHistogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		vector<vector<int>> histogram = calculateHistogram(src);
		showHistogram("Red hist", histogram[2], 255, 255);
		showHistogram("Blue hist", histogram[1], 255, 255);
		showHistogram("Green hist", histogram[0], 255, 255);
		imshow("Source image", src);
	}
}

void compareHistograms() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src1;
		src1 = imread(fname);

		openFileDlg(fname);
		Mat src2;
		src2 = imread(fname);

		vector<vector<int>> h1 = calculateHistogram(src1);
		showHistogram("Red hist", h1[2], 255, 255);
		showHistogram("Blue hist", h1[1], 255, 255);
		showHistogram("Green hist", h1[0], 255, 255);
		
		vector<vector<int>> h2 = calculateHistogram(src2);
		showHistogram("Red hist", h2[2], 255, 255);
		showHistogram("Blue hist", h2[1], 255, 255);
		showHistogram("Green hist", h2[0], 255, 255);

		float cmp = compareHistograms(h1, h2);

		printf("The two histograms are this similar %f\n", cmp);
		imshow("Source image1", src1);
		imshow("Source image2", src2);
	}
}

float compareHistograms(Mat p1, Mat p2) {
		
		vector<vector<int>> h1 = calculateHistogram(p1);
		
		vector<vector<int>> h2 = calculateHistogram(p2);
		
		float cmp = compareHistograms(h1, h2);
	
		return cmp;
}

void BRGtoHSV()
{	
	int cnt = 0;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		imshow("input image", src);
		imshow("HSV image", hsvImg);
		std::string file_name = "DatasetHSV/HSVForeground" + std::to_string(cnt++) + ".jpg";
		imwrite(file_name, hsvImg);
		waitKey();
	}
}

void detectImage() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat src_image = imread(fname);
		cv::String best_match;
		float min_cmp = FLT_MAX;
		vector<cv::String> fn; // std::string in opencv2.4, but cv::String in 3.0
		string folder_name = "C:\\Users\\kadar\\OneDrive\\Desktop\\an3sem2\\Image Processing\\project\\jpg\\DataSet";
		cv::glob(folder_name, fn, false);
		cout << fn.size() << endl;
		for (int i = 0; i < fn.size(); i++) {
			cout << i << endl;
			Mat catalog_img = imread(fn[i]);
			float cmp = compareHistograms(src_image, catalog_img);
			if (cmp != 0 && cmp < min_cmp) {
				min_cmp = cmp;
				best_match = fn[i];
			}

		}

		Mat catalog_img = imread(best_match);
		cout << "Best match is picture " << best_match << "with similarity of " << min_cmp << endl;
		imshow("Best match", catalog_img);
		waitKey();
	}
}

void detectHSVImage() {
		char fname[MAX_PATH];
		while (openFileDlg(fname)) {

			Mat src_image = imread(fname);
			cv::String best_match;
			float min_cmp = FLT_MAX;
			vector<cv::String> fn; // std::string in opencv2.4, but cv::String in 3.0
			string folder_name = "C:\\Users\\kadar\\OneDrive\\Desktop\\an3sem2\\Image Processing\\project\\jpg\\DataSetHSV";
			cv::glob(folder_name, fn, false);
			cout << fn.size() << endl;
			for (int i = 0; i < fn.size(); i++) {
				cout << i << endl;
				Mat catalog_img = imread(fn[i]);
				float cmp = compareHistograms(src_image, catalog_img);
				if (cmp != 0 && cmp < min_cmp) {
					min_cmp = cmp;
					best_match = fn[i];
				}

			}

			Mat catalog_img = imread(best_match);
			cout << "Best match is picture " << best_match << "with similarity of " << min_cmp << endl;
			imshow("Best match", catalog_img);
			waitKey();
		}

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - RGB -> HSV\n");
		printf(" 2 - Segmentation with grabcut\n");
		printf(" 3 - Calculate Histogram\n");
		printf(" 4 - Compare Histograms\n");
		printf(" 5 - Detect Image\n");
		printf(" 6 - Detect HSV Image\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				BRGtoHSV();
				break;
			case 2:
				segmentImage();
				break;
			case 3: 
				calculateHistogram();
				break;
			case 4:
				compareHistograms();
				break;
			case 5:
				detectImage();
				break;
			case 6:
				detectHSVImage();
				
		}
	}
	while (op!=0);
	return 0;
}