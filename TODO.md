- MKL implementation of sparse operations is not working properly.

- Update `ocr.off_line.param` package to use `Matrix<T>` iterators instead of
  direct access to data.

- Update `Image` package to use `Matrix<T>::random_access_iterator`, instead of
  direct access to data. Therefore, image would work with any underlying
  `matrix`, even not simple ones.
