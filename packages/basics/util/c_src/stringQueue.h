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
#ifndef STRINGQUEUE_H
#define STRINGQUEUE_H

class StringQueue {
  struct Node {
    static const int bufferSize = 2048 - 2*sizeof(int) - sizeof(Node*);
    int first,last;
    Node *next;
    char buffer[bufferSize];
    Node() : first(0), last(0), next(0) {}
    int occuped()   const { return last-first; }
    int freeSpace() const { return bufferSize-last; }
    void putCharNoCheck(char c) { buffer[last] = c; last++; }
  };

  Node *firstNode, *lastNode;
  int theSize;
  void checkNonEmpty();
  void checkLast();

public:

  StringQueue();
  ~StringQueue() { clear(); }

  void clear();
  int  size()  const { return theSize; }
  bool empty() const { return size() == 0; }

  void putchar(char);
  void putchars(char c,int times);
  void putBuffer(const char* buffer,int buffersz);
  void putStringQueue(const StringQueue &other);
  void print(const char *str);

  void printEsc(const char *str);
  void printf(const char *fmt, ...);

  // returns a buffer pointer and a buffer size
  int getBufferChunk(char **buffer);

  // drops first bufferAdvance bytes from StringQueue
  void advanceBuffer(int bufferAdvance);

  // returns a new buffer
  char *exportBuffer(int &sz) const;

  // syntactic sugar :)
  StringQueue& operator<< (const char *str) { print(str); return *this; }
  StringQueue& operator<< (int    i) { printf("%d",i); return *this; }
  StringQueue& operator<< (float  f) { printf("%g",f); return *this; }
  StringQueue& operator<< (double f) { printf("%g",f); return *this; }

};

#endif // STRINGQUEUE_H

