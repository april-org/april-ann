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
#include "stringQueue.h"

#include <cstdio>  // vsnprintf
#include <cstring> // memcpy(dest,src,sz)
#include <cstdlib>
#include <cstdarg>

#include "maxmin.h"

namespace AprilUtils {

  StringQueue::StringQueue() {
    firstNode = lastNode = 0;
    theSize   = 0;
  }

  void StringQueue::clear() {
    while (firstNode) {
      Node *aux = firstNode;
      firstNode = firstNode->next;
      delete aux;
    }
    lastNode = 0;
    theSize  = 0;
  }

  inline void StringQueue::checkNonEmpty() {
    if (!lastNode) {
      firstNode = lastNode = new Node;
    }
  }

  inline void StringQueue::checkLast() {
    if (lastNode->last == Node::bufferSize) {
      lastNode->next = new Node;
      lastNode       = lastNode->next;
    }
  }

  void StringQueue::putchar(char c) {
    checkNonEmpty();
    checkLast();
    lastNode->putCharNoCheck(c);
    theSize++;
  }

  void StringQueue::putchars(char c, int times) {
    checkNonEmpty();
    for (int i=0;i<times;++i) {
      checkLast();
      lastNode->putCharNoCheck(c);
    }
    theSize += times;
  }

  void StringQueue::print(const char *str) {
    checkNonEmpty();
    for (char c = *str++; c != '\0'; c = *str++) {
      checkLast();
      lastNode->putCharNoCheck(c);
      theSize++;
    }
  }

  void StringQueue::printEsc(const char *str) {
    checkNonEmpty();
    for (char c = *str++; c != '\0'; c = *str++) {
      bool esc=true;
      switch (c) {
      case '\\':
	if (*str=='u') esc=false; break; // copy \uHHHH verbatim
      case '"':                   break;
      case '\b': c   = 'b';       break;
      case '\f': c   = 'f';       break;
      case '\n': c   = 'n';       break;
      case '\r': c   = 'r';       break;
      case '\t': c   = 't';       break;
      default:   esc = false;     break;
      }
      if (esc) {
	checkLast();
	lastNode->putCharNoCheck('\\');
	theSize++;
      }
      checkLast();
      lastNode->putCharNoCheck(c);
      theSize++;
    }
  }

  void StringQueue::putBuffer(const char* buffer,int buffersz) {  
    if (buffersz <= 0) return;
    checkNonEmpty();
    theSize += buffersz;
    while (buffersz > 0) {
      checkLast();
      int toCopy = min(buffersz,lastNode->freeSpace()); 
      memcpy(lastNode->buffer+lastNode->last, buffer, toCopy);
      lastNode->last += toCopy;
      buffer         += toCopy;
      buffersz       -= toCopy;	   
    }
  }

  void StringQueue::putStringQueue(const StringQueue &other) {
    for (const Node *r = other.firstNode; r!=0; r=r->next)
      putBuffer(r->buffer,r->occuped()); // theSize is incremented there
  }

  void StringQueue::printf(const char *fmt, ...) {
    const int FIRSTSIZE = 256;
    int       bufsize   = FIRSTSIZE;
    char     *buf       = 0;
    for (;;) {

      delete[] buf;
      buf = new char[bufsize];

      va_list args;
      va_start(args, fmt);
      int outsize = vsnprintf(buf, bufsize, fmt, args);
      va_end(args);

      if (outsize == -1) {
        // Clear indication that output was truncated, but no clear
        // indication of how big buffer needs to be, so simply double
        // existing buffer size for next time.
        bufsize *= 2;
      } else if (outsize == bufsize) {
        // Output was truncated (since at least the \0 could not fit),
        // but no indication of how big the buffer needs to be, so just
        // double existing buffer size for next time.
        bufsize *= 2;
      } else if (outsize > bufsize) {
        // Output was truncated, but we were told exactly how big the
        // buffer needs to be next time. Add two chars to the returned
        // size. One for the \0, and one to prevent ambiguity in the
        // next case below.
        bufsize += 2;
      } else if (outsize == bufsize - 1) {
        // This is ambiguous. May mean that the output string exactly
        // fits, but on some systems the output string may have been
        // trucated. We can't tell.  Just double the buffer size for
        // next time.
        bufsize *= 2;
      } else { // Output was not truncated
        putBuffer(buf,outsize); // theSize is incremented there
        delete[] buf;
        return;
      }
    }
  }

  // returns a buffer pointer and a buffer size
  int StringQueue::getBufferChunk(char **buffer) {
    if (!firstNode) { *buffer=0; return 0; }
    *buffer = firstNode->buffer + firstNode->first;
    return (firstNode->last - firstNode->first);
  }

  char* StringQueue::exportBuffer(int &sz) const {
    sz = size();
    char *resul, *dest;
    resul = dest = new char[sz+1];
    for (const Node *r = firstNode; r!=0; r=r->next) {
      int t = r->occuped();
      memcpy(dest, r->buffer, t);
      dest += t;
    }
    *dest = '\0';
    return resul;
  }

  // drops first bufferAdvance bytes from StringQueue
  // a bufferAdvance value <= previous getBufferChunk result is assumed
  void StringQueue::advanceBuffer(int bufferAdvance) {
    firstNode->first += bufferAdvance;
    theSize          -= bufferAdvance;
    if (firstNode->first >= firstNode->last) {
      if (firstNode == lastNode) { // only one node
        firstNode->first = firstNode->last = 0; // reset node
      } else { // firstNode->first == Node::bufferSize and >1 Node
        Node *aux = firstNode;
        firstNode = firstNode->next;
        delete aux;
      }
    }
  }

} // namespace AprilUtils
