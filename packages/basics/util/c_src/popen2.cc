/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#include "popen2.h"
 
#define READ  0
#define WRITE 1

namespace april_utils {

  pid_t popen2(const char *command, int *infp, int *outfp) {
    int p_stdin[2], p_stdout[2];
    pid_t pid;
 
    if (pipe(p_stdin) != 0 || pipe(p_stdout) != 0) return -1;
 
    pid = fork();
 
    if (pid < 0) {
      return pid;
    }
    else if (pid == 0) { // el hijo
      close(p_stdin[WRITE]);
      dup2(p_stdin[READ], READ);
      close(p_stdout[READ]);
      dup2(p_stdout[WRITE], WRITE);
    
      execl("/bin/sh", "sh", "-c", command, NULL);
      perror("execl");
      exit(1);
    }
    close(p_stdin[READ]);
    close(p_stdout[WRITE]);
    *infp  = p_stdin[WRITE];
    *outfp = p_stdout[READ];
  
    return pid;
  }

}
