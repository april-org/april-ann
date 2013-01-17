#include <fcntl.h>
#include <stdio.h>
#include "../c_src/file_stream.cc"

int main()
{
  file_stream fs1("file.txt", O_RDONLY);
  char *line, *all;
  fs1.read_all(&all);
  printf ("ALL:\n%s\n\n", all);
  delete all;
  fs1.closefd();
  
  file_stream fs2("file.txt", O_RDONLY);
  int n;
  int i=0;
  printf ("LINES:\n");
  while ((n=fs2.read_line(&line)) > 0) {
    printf ("%d: len=%d  '%s'\n", i,n,line);
    ++i;
    delete line;
  }
  fs2.closefd();

  file_stream fs3("file.txt", O_RDONLY);
  i=0;
  printf ("\nTOKENS:\n");
  while ((n=fs3.read_token(&line," |\n")) > 0) {
    printf ("%d: len=%d  '%s'\n", i,n,line);
    ++i;
    delete line;
  }
  fs3.closefd();
}
