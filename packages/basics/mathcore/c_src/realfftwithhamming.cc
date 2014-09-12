/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2010, Salvador España-Boquera
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#include <cmath>
#include <cstdio>

#include "realfftwithhamming.h"

namespace AprilMath {

  RealFFTwithHamming::RealFFTwithHamming(int vSize) {
    this->vSize = vSize;
    // obtain first power of 2 >= vSize
    bitsnFFT = 1; nFFT = 2;
    while (nFFT < vSize) {
      nFFT *= 2;
      bitsnFFT++;
    }
    //
    invsqrt2 = 1.0/sqrt(2.0);
    vec = new double[nFFT];
    tmp = new double[nFFT];
    tbl = new nodtblfft[nFFT>>3];
    double aux = M_PI/(nFFT>>1);
    for (int i=0;i<(nFFT>>3);i++) {
      tbl[i].c  = cos(aux*i);
      tbl[i].s  = sin(aux*i);
      tbl[i].c3 = cos(aux*i*3);
      tbl[i].s3 = sin(aux*i*3);
    }
    hamming_window = new double[vSize];
    double factor = 2.0 * M_PI / (vSize - 1.0);
    for(int i=0; i < vSize; i++ )
      hamming_window[i] = 0.54 - 0.46 * cos( factor * i );  
  }

  RealFFTwithHamming::~RealFFTwithHamming() {
    delete[] tmp;
    delete[] vec;
    delete[] tbl;
    delete[] hamming_window;
  }

  /////////////////////////////////////////////////////////
  // Sorensen in-place split-radix FFT for real values
  // Sorensen et al: Real-Valued Fast Fourier Transform Algorithms,
  // IEEE Trans. ASSP, ASSP-35, No. 6, June 1987
  //
  // input vector data is an array of doubles:
  // re(0),re(1),re(2),...,re(vSize-1)
  //
  // output vector data is
  // s(0), s(1), s(nFFT/2-1)
  // where s = abs(fft(data))

  void RealFFTwithHamming::operator() (const double input[], double output[]) {

    int i;
    // copy data to vec applying the Hamming window:
    for (i=0; i<vSize;i++)   vec[i] = input[i] * hamming_window[i];
    for (i=vSize;i<nFFT;i++) vec[i] = 0;
  
    // ----------------  SPLIT  ---------------
    // input vector data is an array of doubles:
    // re(0),re(1),re(2),...,re(nFFT-1)
  
    int    j,k,i5,i6,i7,i8,i0,id,i1,i2,i3,i4,n2,n4,n8,ie,je;
    int halfnFFT = nFFT >> 1;
    double t1,t2,t3,t4,t5,t6,ss1,ss3,cc1,cc3;
  
    n4=nFFT-1;
  
    //data shuffling
    for (i=0,j=0,n2=halfnFFT; i<n4 ; i++) {
      if (i<j) {
	t1     = vec[j];
	vec[j] = vec[i];
	vec[i] = t1;
      }
      k=n2;
      while (k<=j) {
	j -=  k;
	k >>= 1;	
      }
      j += k;
    }
  
    /*----------------------*/
  
    //length two butterflies	
    i0=0;
    id=4;
    do{
      for (; i0<n4; i0+=id){ 
	i1=i0+1;
	t1=vec[i0];
	vec[i0]=t1+vec[i1];
	vec[i1]=t1-vec[i1];
      }
      id<<=1;
      i0=id-2;
      id<<=1;
    } while ( i0<n4 );
  
    /*----------------------*/
    //L shaped butterflies
    n2=2;
    for(k=nFFT;k>2;k>>=1){  
      n2<<=1;
      n4=n2>>2;
      n8=n2>>3;
      ie = je = nFFT/n2;
      i1=0;
      id=n2<<1;
      do{ 
	for (; i1<nFFT; i1+=id){
	  i2=i1+n4;
	  i3=i2+n4;
	  i4=i3+n4;
	  t1=vec[i4]+vec[i3];
	  vec[i4]-=vec[i3];
	  vec[i3]=vec[i1]-t1;
	  vec[i1]+=t1;
	  if (n4!=1){
	    i0=i1+n8;
	    i2+=n8;
	    i3+=n8;
	    i4+=n8;
	    t1=(vec[i3]+vec[i4])*invsqrt2;
	    t2=(vec[i3]-vec[i4])*invsqrt2;
	    vec[i4]=vec[i2]-t1;
	    vec[i3]=-vec[i2]-t1;
	    vec[i2]=vec[i0]-t2;
	    vec[i0]+=t2;
	  }
	}
	id<<=1;
	i1=id-n2;
	id<<=1;
      } while ( i1<nFFT );
      for (j=2; j<=n8; j++){  
	cc1=tbl[je].c;
	ss1=tbl[je].s;
	cc3=tbl[je].c3;
	ss3=tbl[je].s3;
	je += ie;
	i=0;
	id=n2<<1;
	do{
	  for (; i<nFFT; i+=id){  
	    i1=i+j-1;
	    i2=i1+n4;
	    i3=i2+n4;
	    i4=i3+n4;
	    i5=i+n4-j+1;
	    i6=i5+n4;
	    i7=i6+n4;
	    i8=i7+n4;
	    t1=vec[i3]*cc1+vec[i7]*ss1;
	    t2=vec[i7]*cc1-vec[i3]*ss1;
	    t3=vec[i4]*cc3+vec[i8]*ss3;
	    t4=vec[i8]*cc3-vec[i4]*ss3;
	    t5=t1+t3;
	    t6=t2+t4;
	    t3=t1-t3;
	    t4=t2-t4;
	    t2=vec[i6]+t6;
	    vec[i3]=t6-vec[i6];
	    vec[i8]=t2;
	    t2=vec[i2]-t3;
	    vec[i7]=-vec[i2]-t3;
	    vec[i4]=t2;
	    t1=vec[i1]+t5;
	    vec[i6]=vec[i1]-t5;
	    vec[i1]=t1;
	    t1=vec[i5]+t4;
	    vec[i5]-=t4;
	    vec[i2]=t1;
	  }
	  id<<=1;
	  i=id-n2;
	  id<<=1;
	} while(i<nFFT);
      }
    }
    // end of split part
  
    //unshuffling - not in-place
    tmp[0]=vec[0];
    tmp[1]=vec[halfnFFT];
    for(i=1;i<halfnFFT;i++) {tmp[i+i]=vec[i];tmp[i+i+1]=vec[nFFT-i];}

    // after unshuffling, tmp contains:
    // re(0),re(vSize/2),re(1),im(1),re(2),im(2),...,re(vSize/2-1),im(vSize/2-1)
  
    output[0] = fabsf(tmp[0]);
    double *aux;
    for (i=1,aux=tmp+2;i<halfnFFT;i++) {
      //output[i] = sqrt(aux[0]*aux[0]+aux[1]*aux[1]);
      // we compute the square:
      output[i] = aux[0]*aux[0]+aux[1]*aux[1];
      aux += 2;
    }
    // output vector data is
    // s(0), s(1), s(nFFT/2-1)
  }

} // namespace AprilMath
