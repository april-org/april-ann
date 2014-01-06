- Update `Image` package to use `Matrix<T>::random_access_iterator`, instead of
  direct access to data. Therefore, image would work with any underlying
  `matrix`, even not simple ones.
