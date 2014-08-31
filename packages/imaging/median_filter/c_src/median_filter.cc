#include "median_filter.h"
#include "ctmf.h"
#include "clamp.h"

using AprilUtils::clamp;

namespace Imaging {

  ImageFloat* medianFilter(ImageFloat *img, int radio) {
    int numpixels = img->width() * img->height();
    unsigned char *src = new unsigned char[numpixels];
    unsigned char *dst = new unsigned char[numpixels];
    unsigned char *r   = src;

    // copiamos la imagen a data:
    for (int y=0;y<img->height();++y)
      for (int x=0;x<img->width();++x)
        *r++ = (unsigned char)(clamp((*img)(x,y), 0.0f, 1.0f)*255.0f);

    // llamamos a la funcion
    ctmf(src,dst,img->width(),img->height(),
         img->width(),img->width(),
         radio,1,512*1024);

    // escribimos el resultado en una imagen nueva
    int dims[2];
    dims[0] = img->height();
    dims[1] = img->width();
    Basics::MatrixFloat *mat = new Basics::MatrixFloat(2,dims);
    ImageFloat  *resul = new ImageFloat(mat);

    int i=0;
    for (Basics::MatrixFloat::iterator it(mat->begin());
         it!=mat->end(); ++it, ++i)
      *it = dst[i]/255.0f;
  
    delete[] src;
    delete[] dst;

    return resul;
  }

} // namespace Imaging
