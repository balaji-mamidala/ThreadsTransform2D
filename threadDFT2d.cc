// Threaded two-dimensional Discrete FFT transform
// Balaji Mamidala
// ECE8893 Project 2


#include <iostream>
#include <string>
#include <math.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Bit_Rev_Image(void);
void Transpose(void);
void Conjugate(void);
void Divide_by_N(void);
void Truncate(void);

// Data structure for image data, width and heigth
const char* ImagePath;
Complex *ImageData;
int ImageHeight;
int ImageWidth;
Complex *Twiddle;

void Generate_Twiddle(int N);

// Each thread needs to know how many threads there are.
int nThreads = 17; // 16 helper threads and 1 thread to transpose and collect data

// The mutex and condition variables allow the main thread to know when all helper threads are completed.
pthread_mutex_t TcountMutex;
pthread_mutex_t exitMutex;
pthread_cond_t exitCond;
int Tcount;


// Variables and mutexes for Barrier
int Bcount; // Number of threads presently in the barrier
int FetchAndDecrementCount();
pthread_mutex_t BcountMutex;
bool *localSense; // We will create an array of bools, one per thread
bool globalSense; // Global sense


// Test variables and mutex for pthreads
int Count;
pthread_mutex_t Count_Mutex;


//Generates all the Twiddle factors
void Generate_Twiddle(int N)
{
  Twiddle = new Complex[N/2];
  
  for(int n=0; n < (N/2); n++)
  {
    Twiddle[n].real = cos(2*M_PI*n/N);
    Twiddle[n].imag = -sin(2*M_PI*n/N); 
  }
}

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v, int N)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main
void MyBarrier_Init()// you will likely need some parameters)
{
  // Initialize the mutex used for FetchAndIncrement
  pthread_mutex_init(&BcountMutex, 0);
 
  // Initialize number of threads in barrier
  Bcount = nThreads;
 
  // Create and initialize the localSense arrar, 1 entry per thread
  localSense = new bool[nThreads];
  
  for (int i = 0; i < nThreads; ++i) localSense[i] = true;
  
  // Initialize global sense
  globalSense = true;
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(int myId) // Again likely need parameters
{
  localSense[myId] = !localSense[myId]; // Toggle private sense variable

  if (FetchAndDecrementCount() == 1)
  { 
    // All threads here, reset count and toggle global sense
    Bcount = nThreads;
    globalSense = localSense[myId];
  }

  else
  {
    while (globalSense != localSense[myId]); // Spin
  }
}


int FetchAndDecrementCount()
{ 
  // We don’t have an atomic FetchAndDecrement, but we can get the same behavior by using a mutex
  pthread_mutex_lock(&BcountMutex);
  int myCount = Bcount;
  Bcount--;
  pthread_mutex_unlock(&BcountMutex);
  return myCount;
}
                    
void Transform1D(Complex* h, int N)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)
  
  int Np; // Calculating Np point DFT 
  int w; // No of times Np is executed. No of blocks on which Np DFT needs to be executed
  int k; // No of equations 

  for(Np=2; Np<=N; Np*=2)
  {
    for(w=0; w<(N/Np); w++)
    {
      for(k=0; k<(Np/2); k++)
      {
        Complex temp = h[w*Np + k];
        h[w*Np + k] = temp + Twiddle[k*N/Np] * h[w*Np + k + (Np/2)]; //H[0] = H[0] + W[0]*H[1]
        h[w*Np + k + (Np/2)] = temp - Twiddle[k*N/Np] * h[w*Np + k + (Np/2)]; //H[1] = H[0] - W[0]*H[1]
      }
    }
  }
}

void* Transform2DTHread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
  
  int myId = (unsigned long)v;
 
  // Wait for Master thread to get image data and re-order
  MyBarrier(myId);

  //sRow is the first row to compute DFT for the helper thread 
  //eRow is last row to compute DFT for the helper thread
  //Row_Cpu is the rows per helper thread
  //Assumption: Number of rows and columns is exactly divisible by number of helper threads
  int sRow, eRow, Row_Cpu = 0;
  int nRows = ImageHeight;

  Row_Cpu = nRows/(nThreads-1);
  sRow = Row_Cpu * (myId-1); // Helper thread ID starts from 1
  eRow = sRow + Row_Cpu - 1; 


  int h;
  //Calculate 1d dft for the rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D( (ImageData+(h*ImageWidth)) , ImageWidth);
  }
    
  //Enter barrier to indicate to master thread that 1D-DFT is done
  MyBarrier(myId);
  

  // Wait for Master thread to do Transpose and re-order
  MyBarrier(myId);
  
  //Calculate 1d dft again for the rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D( (ImageData+(h*ImageWidth)) , ImageWidth);
  }

  // Enter barrier and to indicate to master thread that 2D-DFT is done
  MyBarrier(myId);

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////


  // Wait for Master thread to do transpose, conjugate, re-order
  MyBarrier(myId);
  
  //Calculate 1d dft for the rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D( (ImageData+(h*ImageWidth)) , ImageWidth);
  }

  // Enter barrier and to indicate to master thread that 1D-DFT is done
  MyBarrier(myId);

  // Wait for Master thread to finish to do conjugate, transpose, conjugate and re-order
  MyBarrier(myId);
  
  //Calculate 2d inverse dft for the rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D( (ImageData+(h*ImageWidth)) , ImageWidth);
  }

  // Enter barrier and to indiacte to master thread that 2D-DFT is done
  MyBarrier(myId);

  //This thread is done; decrement the active count and see if all have finished
  pthread_mutex_lock(&TcountMutex);
  //cout << "My ID:" << myId << endl;
  //cout << "Start row: " << sRow << endl;
  //cout << "End Row: " << eRow << endl;
  Tcount--;
  if (Tcount == 0)
  { 
    // Last to exit, notify main
    pthread_mutex_unlock(&TcountMutex);
    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  else
  {
     pthread_mutex_unlock(&TcountMutex);
  }  

  return 0;
}


//void Transform2D(const char* inputFN) 
void* Transform2D(void* v) 
{ // Do the 2D transform here.
  //InputImage image(inputFN);  // Create the helper object for reading the image
  // Create the global pointer to the image array data
  // Create 16 threads
  // Wait for all threads complete
  // Write the transformed data

  int myId = (unsigned long)v;  
  
  // Initialize and create data structure for image
  InputImage image(ImagePath);
  ImageHeight = image.GetHeight();
  ImageWidth = image.GetWidth();
  
  ImageData = new Complex[ImageHeight * ImageWidth]; 
  ImageData = image.GetImageData();

  Generate_Twiddle(ImageWidth);
   
  
  // Use bit reversal to organize the entire image data in correct format so that helper threads can do 1D-DFT
  Bit_Rev_Image();  

  // Enter barrier to indicate to the helper threads that organizing of image data is done
  MyBarrier(myId); 

  // Enter barrier and wait for helper threads to finish 1D-DFT
  MyBarrier(myId);

  image.SaveImageData("MyAfter1d.txt", ImageData, ImageWidth, ImageHeight); 

  // Transpose the data so that 1D-DFT can be done on the columns
  Transpose();

  // Use bit reversal to organize the entire image data in correct format so that helper threads can do 1D-DFT
  Bit_Rev_Image();  

  // Enter barrier to indicate to the helper threads that organizing of image data is done
  MyBarrier(myId); 

  // Enter barrier and wait for helper threads to finish 2D-DFT
  MyBarrier(myId);
  
  Transpose();
    
  image.SaveImageData("MyAfter2d.txt", ImageData, ImageWidth, ImageHeight);
  
  Conjugate();
 
  Bit_Rev_Image();
  
  // Enter barrier to indicate to the helper threads that data organizing is done  
  MyBarrier(myId);

  // Wait for helper threads to do 1D-DFT
  MyBarrier(myId); 

  Divide_by_N();

  Conjugate();
 
  Transpose();

  Conjugate();

  Bit_Rev_Image();

  // Enter barrier to indicate to the helper threads that data organizing is done
  MyBarrier(myId);
 
  // Wait for helper threads to do 1D-DFT
  MyBarrier(myId);

  Divide_by_N();

  Conjugate();
  
  Transpose();

  Truncate();

  image.SaveImageData("MyAfterInverse.txt", ImageData, ImageWidth, ImageHeight);
 
  //Delete pointers
  //delete(ImagePath);
  delete(ImageData);
  delete(Twiddle);
 
  // This thread is done; decrement the active count and see if all have finished
  pthread_mutex_lock(&TcountMutex);
  Tcount--;
  if (Tcount == 0)
  { 
    // Last to exit, notify main
    pthread_mutex_unlock(&TcountMutex);
    pthread_mutex_lock(&exitMutex);
    pthread_cond_signal(&exitCond);
    pthread_mutex_unlock(&exitMutex);
  }
  else
  {
     pthread_mutex_unlock(&TcountMutex);
  }  
 
  return 0; 
}


void Bit_Rev_Image(void)
{

  int w,h;
  int rev_w;
  for(h=0; h<ImageHeight; h++)
  {
    //Use bit reversal to get data in the correct format for the row
    for(w=0; w<ImageWidth; w++)
    { 
      rev_w = ReverseBits(w, ImageWidth); 
      if(rev_w > w)
      {
        Complex temp = ImageData[h*ImageWidth + w];
        ImageData[h*ImageWidth + w] = ImageData[h*ImageWidth + rev_w];
        ImageData[h*ImageWidth + rev_w] = temp;
      }
    }
  }
}


void Transpose(void)
{
  int r,c;
  for(r=0; r< ImageHeight; r++)
  {
    for(c=0; c<r; c++)
    {
      Complex temp =  ImageData[(c*ImageHeight) + r];
      ImageData[(c*ImageHeight) + r] =  ImageData[(r*ImageWidth) + c];
      ImageData[(r*ImageWidth) + c] = temp;
    }
  }
}


void Conjugate(void)
{
  int i;
  for(i=0; i<(ImageHeight*ImageWidth); i++)
  {
    ImageData[i] = ImageData[i].Conj();
  }
}


void Divide_by_N(void)
{
  int i;
  for(i=0; i<(ImageHeight*ImageWidth); i++)
  {
    ImageData[i].real /= ImageWidth;
    ImageData[i].imag /= ImageWidth;
  }
}


void Truncate(void)
{
  int r,c;

  for(r=0; r< ImageHeight; r++)
  {
    for(c=0; c<ImageWidth; c++)
    {
      if(ImageData[(r*ImageHeight) + c].Mag().real < (double)0.1)
      {
        ImageData[(r*ImageHeight) + c].real = 0;
        ImageData[(r*ImageHeight) + c].imag = 0;
      }
   }
  }

}


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line 

  // Initialize path for image
  ImagePath = fn.c_str(); 

  // All mutex and condition variables must be "initialized"
  pthread_mutex_init(&exitMutex,0);
  pthread_mutex_init(&TcountMutex,0);
  pthread_cond_init(&exitCond, 0);
  
  // Initialize the barrier for threads
  MyBarrier_Init();
 
  // Main holds the exit mutex until waiting for exitCond condition
  pthread_mutex_lock(&exitMutex);

  Tcount = nThreads; // Total threads (to be) started
  
  int i = 0;
  // Create master thread that does array transpose
  pthread_t pt; // pThread variable (output param from create)
  // Third param is the thread starting function. Fourth param is passed to the thread starting function
  pthread_create(&pt, 0, Transform2D, (void*)i);
  
  // Now start 16 helper threads
  for (i = 1; i < nThreads; ++i)
  {
    // Third param is the thread starting function. Fourth param is passed to the thread starting function
    pthread_create(&pt, 0, Transform2DTHread, (void*)i);
  }

  // Main program now waits until all child threads completed
  pthread_cond_wait(&exitCond, &exitMutex);




  //Code for computing DFT without threads
  /*InputImage image(fn.c_str());
  ImageHeight = image.GetHeight();
  ImageWidth = image.GetWidth();
  
  ImageData = new Complex[ImageHeight * ImageWidth]; 
  ImageData = image.GetImageData();

  Generate_Twiddle(ImageWidth);

  int w,h;
  int rev_w;
  for(h=0; h<ImageHeight; h++)
  {
    //Use bit reversal to get data in the correct format
    for(w=0; w<ImageWidth; w++)
    { 
      rev_w = ReverseBits(w, ImageWidth); 
      if(rev_w > w)
      {
        Complex temp = ImageData[h*ImageWidth + w];
        ImageData[h*ImageWidth + w] = ImageData[h*ImageWidth + rev_w];
        ImageData[h*ImageWidth + rev_w] = temp;
      }
    }

    Transform1D(ImageData+(h*ImageWidth), ImageWidth);
  }

  image.SaveImageData("MyAfter1d.txt", ImageData, ImageWidth, ImageHeight);
  */
}  
  
