/*
** $Id: lua.c,v 1.206 2012/09/29 20:07:06 roberto Exp $
** Lua stand-alone interpreter
** See Copyright Notice in lua.h
*/


#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define lua_c

#include "lua.h"

#include "lauxlib.h"
#include "lualib.h"


#if !defined(LUA_PROMPT)
#define LUA_PROMPT		"> "
#define LUA_PROMPT2		">> "
#endif

#if !defined(LUA_PROGNAME)
#define LUA_PROGNAME		"lua"
#endif

#if !defined(LUA_MAXINPUT)
#define LUA_MAXINPUT		512
#endif

#if !defined(LUA_INIT)
#define LUA_INIT		"LUA_INIT"
#endif

#define LUA_INITVERSION  \
	LUA_INIT "_" LUA_VERSION_MAJOR "_" LUA_VERSION_MINOR


/*
** lua_stdin_is_tty detects whether the standard input is a 'tty' (that
** is, whether we're running lua interactively).
*/
#if defined(LUA_USE_ISATTY)
#include <unistd.h>
#define lua_stdin_is_tty()	isatty(0)
#elif defined(LUA_WIN)
#include <io.h>
#include <stdio.h>
#define lua_stdin_is_tty()	_isatty(_fileno(stdin))
#else
#define lua_stdin_is_tty()	1  /* assume stdin is a tty */
#endif


/*
** lua_readline defines how to show a prompt and then read a line from
** the standard input.
** lua_saveline defines how to "save" a read line in a "history".
** lua_freeline defines how to free a line read by lua_readline.
*/
#if defined(LUA_USE_READLINE)

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>
#define lua_readline(L,b,p)	((void)L, ((b)=readline(p)) != NULL)
#define lua_saveline(L,idx) \
        if (lua_rawlen(L,idx) > 0)  /* non-empty line? */ \
          add_history(lua_tostring(L, idx));  /* add it to history */
#define lua_freeline(L,b)	((void)L, free(b))

#elif !defined(lua_readline)

#define lua_readline(L,b,p) \
        ((void)L, fputs(p, stdout), fflush(stdout),  /* show prompt */ \
        fgets(b, LUA_MAXINPUT, stdin) != NULL)  /* get line */
#define lua_saveline(L,idx)	{ (void)L; (void)idx; }
#define lua_freeline(L,b)	{ (void)L; (void)b; }

#endif




lua_State *globalL = NULL;

static const char *progname = LUA_PROGNAME;



static void lstop (lua_State *L, lua_Debug *ar) {
  (void)ar;  /* unused arg. */
  lua_sethook(L, NULL, 0, 0);
  luaL_error(L, "interrupted!");
}


static void laction (int i) {
  signal(i, SIG_DFL); /* if another SIGINT happens before lstop,
                              terminate process (default action) */
  lua_sethook(globalL, lstop, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, 1);
}


static void print_usage (const char *badoption) {
  luai_writestringerror("%s: ", progname);
  if (badoption[1] == 'e' || badoption[1] == 'l')
    luai_writestringerror("'%s' needs argument\n", badoption);
  else
    luai_writestringerror("unrecognized option '%s'\n", badoption);
  luai_writestringerror(
  "usage: %s [options] [script [args]]\n"
  "Available options are:\n"
  "  -e stat  execute string " LUA_QL("stat") "\n"
  "  -i       enter interactive mode after executing " LUA_QL("script") "\n"
  "  -l name  require library " LUA_QL("name") "\n"
  "  -v       show version information\n"
  "  -E       ignore environment variables\n"
  "  --       stop handling options\n"
  "  -        stop handling options and execute stdin\n"
  ,
  progname);
}


static void l_message (const char *pname, const char *msg) {
  if (pname) luai_writestringerror("%s: ", pname);
  luai_writestringerror("%s\n", msg);
}


static int report (lua_State *L, int status) {
  if (status != LUA_OK && !lua_isnil(L, -1)) {
    const char *msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error object is not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
    /* force a complete garbage collection in case of errors */
    lua_gc(L, LUA_GCCOLLECT, 0);
  }
  return status;
}


/* the next function is called unprotected, so it must avoid errors */
static void finalreport (lua_State *L, int status) {
  if (status != LUA_OK) {
    const char *msg = (lua_type(L, -1) == LUA_TSTRING) ? lua_tostring(L, -1)
                                                       : NULL;
    if (msg == NULL) msg = "(error object is not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
}


static int traceback (lua_State *L) {
  const char *msg = lua_tostring(L, 1);
  if (msg)
    luaL_traceback(L, L, msg, 1);
  else if (!lua_isnoneornil(L, 1)) {  /* is there an error object? */
    if (!luaL_callmeta(L, 1, "__tostring"))  /* try its 'tostring' metamethod */
      lua_pushliteral(L, "(no error message)");
  }
  return 1;
}


static int docall (lua_State *L, int narg, int nres) {
  int status;
  int base = lua_gettop(L) - narg;  /* function index */
  lua_pushcfunction(L, traceback);  /* push traceback function */
  lua_insert(L, base);  /* put it under chunk and args */
  globalL = L;  /* to be available to 'laction' */
  signal(SIGINT, laction);
  status = lua_pcall(L, narg, nres, base);
  signal(SIGINT, SIG_DFL);
  lua_remove(L, base);  /* remove traceback function */
  return status;
}


static void print_version (void) {
  luai_writestring(LUA_COPYRIGHT, strlen(LUA_COPYRIGHT));
  luai_writeline();
}


static int getargs (lua_State *L, char **argv, int n) {
  int narg;
  int i;
  int argc = 0;
  while (argv[argc]) argc++;  /* count total number of arguments */
  narg = argc - (n + 1);  /* number of arguments to the script */
  luaL_checkstack(L, narg + 3, "too many arguments to script");
  for (i=n+1; i < argc; i++)
    lua_pushstring(L, argv[i]);
  lua_createtable(L, narg, n + 1);
  for (i=0; i < argc; i++) {
    lua_pushstring(L, argv[i]);
    lua_rawseti(L, -2, i - n);
  }
  return narg;
}


static int dofile (lua_State *L, const char *name) {
  int status = luaL_loadfile(L, name);
  if (status == LUA_OK) status = docall(L, 0, 0);
  return report(L, status);
}


static int dostring (lua_State *L, const char *s, const char *name) {
  int status = luaL_loadbuffer(L, s, strlen(s), name);
  if (status == LUA_OK) status = docall(L, 0, 0);
  return report(L, status);
}


static int dolibrary (lua_State *L, const char *name) {
  int status;
  lua_getglobal(L, "require");
  lua_pushstring(L, name);
  status = docall(L, 1, 1);  /* call 'require(name)' */
  if (status == LUA_OK)
    lua_setglobal(L, name);  /* global[name] = require return */
  return report(L, status);
}

/* ------------------------------------------------------------------------ */

#ifdef LUA_USE_READLINE
/*
** Lua 5.2.0 advanced readline support for the GNU readline and history
** libraries or compatible replacements.
**
** Author: Mike Pall.
** Maintainer: Sean Bolton (sean at smbolton dot com).
**
** Copyright (C) 2004-2006, 2011 Mike Pall. Same license as Lua. See lua.h.
**
** Advanced features:
** - Completion of keywords and global variable names.
** - Recursive and metatable-aware completion of variable names.
** - Context sensitive delimiter completion.
** - Save/restore of the history to/from a file (LUA_HISTORY env variable).
** - Setting a limit for the size of the history (LUA_HISTSIZE env variable).
** - Setting the app name to allow for $if lua ... $endif in ~/.inputrc.
**
** Start lua and try these (replace ~ with the TAB key):
**
** ~~
** fu~foo() ret~fa~end<CR>
** io~~~s~~~o~~~w~"foo\n")<CR>
**
** The ~~ are just for demonstration purposes (io~s~o~w~ suffices, of course).
**
** If you are used to zsh/tcsh-style completion support, try adding
** 'TAB: menu-complete' and 'C-d: possible-completions' to your ~/.inputrc.
**
** The patch has been successfully tested with:
**
** GNU    readline 2.2.1  (1998-07-17)
** GNU    readline 4.0    (1999-02-18) [harmless compiler warning]
** GNU    readline 4.3    (2002-07-16)
** GNU    readline 5.0    (2004-07-27)
** GNU    readline 5.1    (2005-12-07)
** GNU    readline 5.2    (2006-10-11)
** GNU    readline 6.0    (2009-02-20)
** GNU    readline 6.2    (2011-02-13)
** MacOSX libedit  2.11   (2008-07-12)
** NETBSD libedit  2.6.5  (2002-03-25)
** NETBSD libedit  2.6.9  (2004-05-01)
**
** Change Log:
** 2004-2006  Mike Pall   - original patch
** 2009/08/24 Sean Bolton - updated for GNU readline version 6
** 2011/11/15 Sean Bolton - updated for Lua 5.2
** 2011/12/14 Sean Bolton - fixed segfault when using Mac OS X libedit 2.11
*/

#include <ctype.h>

static char *lua_rl_hist;
static int lua_rl_histsize;

static lua_State *lua_rl_L;  /* User data is not passed to rl callbacks. */

/* Reserved keywords. */
static const char *const lua_rl_keywords[] = {
  "and", "break", "do", "else", "elseif", "end", "false",
  "for", "function", "goto", "if", "in", "local", "nil", "not", "or",
  "repeat", "return", "then", "true", "until", "while", NULL
};

static int valididentifier(const char *s)
{
  if (!(isalpha(*s) || *s == '_')) return 0;
  for (s++; *s; s++) if (!(isalpha(*s) || isdigit(*s) || *s == '_')) return 0;
  return 1;
}

/* Dynamically resizable match list. */
typedef struct {
  char **list;
  size_t idx, allocated, matchlen;
} dmlist;

/* Add prefix + string + suffix to list and compute common prefix. */
static int lua_rl_dmadd(dmlist *ml, const char *p, size_t pn, const char *s,
			int suf)
{
  char *t = NULL;

  if (ml->idx+1 >= ml->allocated &&
      !(ml->list = (char**)realloc(ml->list, sizeof(char *)*(ml->allocated += 32))))
    return -1;

  if (s) {
    size_t n = strlen(s);
    if (!(t = (char *)malloc(sizeof(char)*(pn+n+(suf?2:1))))) return 1;
    memcpy(t, p, pn);
    memcpy(t+pn, s, n);
    n += pn;
    t[n] = suf;
    if (suf) t[++n] = '\0';

    if (ml->idx == 0) {
      ml->matchlen = n;
    } else {
      size_t i;
      for (i = 0; i < ml->matchlen && i < n && ml->list[1][i] == t[i]; i++) ;
      ml->matchlen = i;  /* Set matchlen to common prefix. */
    }
  }

  ml->list[++ml->idx] = t;
  return 0;
}

/* Get __index field of metatable of object on top of stack. */
static int lua_rl_getmetaindex(lua_State *L)
{
  if (!lua_getmetatable(L, -1)) { lua_pop(L, 1); return 0; }
  lua_pushstring(L, "__index");
  lua_rawget(L, -2); if (lua_isfunction(L, -1)) { lua_pop(L, 1); lua_pushstring(L, "index_table"); lua_rawget(L, -2); }
  lua_replace(L, -2);
  if (lua_isnil(L, -1) || lua_rawequal(L, -1, -2)) { lua_pop(L, 2); return 0; }
  lua_replace(L, -2);
  return 1;
}  /* 1: obj -- val, 0: obj -- */

/* Get field from object on top of stack. Avoid calling metamethods. */
static int lua_rl_getfield(lua_State *L, const char *s, size_t n)
{
  int i = 20;  /* Avoid infinite metatable loops. */
  do {
    if (lua_istable(L, -1)) {
      lua_pushlstring(L, s, n);
      lua_rawget(L, -2);
      if (!lua_isnil(L, -1)) { lua_replace(L, -2); return 1; }
      lua_pop(L, 1);
    }
  } while (--i > 0 && lua_rl_getmetaindex(L));
  lua_pop(L, 1);
  return 0;
}  /* 1: obj -- val, 0: obj -- */

/* Completion callback. */
static char **lua_rl_complete(const char *text, int start, int end)
{
  lua_State *L = lua_rl_L;
  dmlist ml;
  const char *s;
  size_t i, n, dot, loop;
  int savetop;

  if (!(text[0] == '\0' || isalpha(text[0]) || text[0] == '_')) return NULL;

  ml.list = NULL;
  ml.idx = ml.allocated = ml.matchlen = 0;

  savetop = lua_gettop(L);
  lua_pushglobaltable(L);
  for (n = (size_t)(end-start), i = dot = 0; i < n; i++)
    if (text[i] == '.' || text[i] == ':') {
      if (!lua_rl_getfield(L, text+dot, i-dot))
	goto error;  /* Invalid prefix. */
      dot = i+1;  /* Points to first char after dot/colon. */
    }

  /* Add all matches against keywords if there is no dot/colon. */
  if (dot == 0)
    for (i = 0; (s = lua_rl_keywords[i]) != NULL; i++)
      if (!strncmp(s, text, n) && lua_rl_dmadd(&ml, NULL, 0, s, ' '))
	goto error;

  /* Add all valid matches from all tables/metatables. */
  loop = 0;  /* Avoid infinite metatable loops. */
  lua_pushglobaltable(L);
  lua_insert(L, -2);
  do {
    if (lua_istable(L, -1) &&
	(loop == 0 || !lua_rawequal(L, -1, -2)))
      for (lua_pushnil(L); lua_next(L, -2); lua_pop(L, 1))
	if (lua_type(L, -2) == LUA_TSTRING) {
	  s = lua_tostring(L, -2);
	  /* Only match names starting with '_' if explicitly requested. */
	  if (!strncmp(s, text+dot, n-dot) && valididentifier(s) &&
	      (*s != '_' || text[dot] == '_')) {
	    int suf = ' ';  /* Default suffix is a space. */
	    switch (lua_type(L, -1)) {
	    case LUA_TTABLE:    suf = '\0'; break;  /* No way to guess ':'. */
	    case LUA_TFUNCTION:	suf = '(';  break;
	    case LUA_TUSERDATA:
	      if (lua_getmetatable(L, -1)) { lua_pop(L, 1); suf = ':'; }
	      break;
	    }
	    if (lua_rl_dmadd(&ml, text, dot, s, suf)) goto error;
	  }
	}
  } while (++loop < 20 && lua_rl_getmetaindex(L));
  lua_pop(L, 1);

  if (ml.idx == 0) {
error:
    lua_settop(L, savetop);
    return NULL;
  } else {
    /* list[0] holds the common prefix of all matches (may be ""). */
    /* If there is only one match, list[0] and list[1] will be the same. */
    if (!(ml.list[0] = (char *)malloc(sizeof(char)*(ml.matchlen+1))))
      goto error;
    memcpy(ml.list[0], ml.list[1], ml.matchlen);
    ml.list[0][ml.matchlen] = '\0';
    /* Add the NULL list terminator. */
    if (lua_rl_dmadd(&ml, NULL, 0, NULL, 0)) goto error;
  }

  lua_settop(L, savetop);
#if RL_READLINE_VERSION >= 0x0600
  rl_completion_suppress_append = 1;
#endif
  return ml.list;
}

/* Initialize readline library. */
static void lua_rl_init(lua_State *L)
{
  char *s;

  lua_rl_L = L;

  /* This allows for $if lua ... $endif in ~/.inputrc. */
  rl_readline_name = (char*)"lua";
  /* Break words at every non-identifier character except '.' and ':'. */
  rl_completer_word_break_characters = 
    (char*)"\t\r\n !\"#$%&'()*+,-/;<=>?@[\\]^`{|}~";
  rl_completer_quote_characters = (char*)"\"'";
#if RL_READLINE_VERSION < 0x0600
  rl_completion_append_character = '\0';
#endif
  rl_attempted_completion_function = lua_rl_complete;
  rl_initialize();

  /* Start using history, optionally set history size and load history file. */
  using_history();
  if ((s = getenv("LUA_HISTSIZE")) &&
      (lua_rl_histsize = atoi(s))) stifle_history(lua_rl_histsize);
  if ((lua_rl_hist = getenv("LUA_HISTORY"))) read_history(lua_rl_hist);
}

/* Finalize readline library. */
static void lua_rl_exit(lua_State *L)
{
  (void)(L); /* Optionally save history file. */
  if (lua_rl_hist) write_history(lua_rl_hist);
}
#else
#define lua_rl_init(L)		((void)L)
#define lua_rl_exit(L)		((void)L)
#endif

/* ------------------------------------------------------------------------ */

static const char *get_prompt (lua_State *L, int firstline) {
  const char *p;
  lua_getglobal(L, firstline ? "_PROMPT" : "_PROMPT2");
  p = lua_tostring(L, -1);
  if (p == NULL) p = (firstline ? LUA_PROMPT : LUA_PROMPT2);
  return p;
}

/* mark in error messages for incomplete statements */
#define EOFMARK		"<eof>"
#define marklen		(sizeof(EOFMARK)/sizeof(char) - 1)

static int incomplete (lua_State *L, int status) {
  if (status == LUA_ERRSYNTAX) {
    size_t lmsg;
    const char *msg = lua_tolstring(L, -1, &lmsg);
    if (lmsg >= marklen && strcmp(msg + lmsg - marklen, EOFMARK) == 0) {
      lua_pop(L, 1);
      return 1;
    }
  }
  return 0;  /* else... */
}


static int pushline (lua_State *L, int firstline) {
  char buffer[LUA_MAXINPUT];
  char *b = buffer;
  size_t l;
  const char *prmt = get_prompt(L, firstline);
  int readstatus = lua_readline(L, b, prmt);
  lua_pop(L, 1);  /* remove result from 'get_prompt' */
  if (readstatus == 0)
    return 0;  /* no input */
  l = strlen(b);
  if (l > 0 && b[l-1] == '\n')  /* line ends with newline? */
    b[l-1] = '\0';  /* remove it */
  if (firstline && b[0] == '=')  /* first line starts with `=' ? */
    lua_pushfstring(L, "return %s", b+1);  /* change it to `return' */
  else
    lua_pushstring(L, b);
  lua_freeline(L, b);
  return 1;
}


static int loadline (lua_State *L) {
  int status;
  lua_settop(L, 0);
  if (!pushline(L, 1))
    return -1;  /* no input */
  for (;;) {  /* repeat until gets a complete line */
    size_t l;
    const char *line = lua_tolstring(L, 1, &l);
    status = luaL_loadbuffer(L, line, l, "=stdin");
    if (!incomplete(L, status)) break;  /* cannot try to add lines? */
    if (!pushline(L, 0))  /* no more input? */
      return -1;
    lua_pushliteral(L, "\n");  /* add a new line... */
    lua_insert(L, -2);  /* ...between the two lines */
    lua_concat(L, 3);  /* join them */
  }
  lua_saveline(L, 1);
  lua_remove(L, 1);  /* remove line */
  return status;
}


static void dotty (lua_State *L) {
  int status;
  const char *oldprogname = progname;
  progname = NULL;
  lua_rl_init(L);
  while ((status = loadline(L)) != -1) {
    if (status == LUA_OK) status = docall(L, 0, LUA_MULTRET);
    report(L, status);
    if (status == LUA_OK && lua_gettop(L) > 0) {  /* any result to print? */
      luaL_checkstack(L, LUA_MINSTACK, "too many results to print");
      lua_getglobal(L, "print");
      lua_insert(L, 1);
      if (lua_pcall(L, lua_gettop(L)-1, 0, 0) != LUA_OK)
        l_message(progname, lua_pushfstring(L,
                               "error calling " LUA_QL("print") " (%s)",
                               lua_tostring(L, -1)));
    }
  }
  lua_settop(L, 0);  /* clear stack */
  luai_writeline();
  lua_rl_exit(L);
  progname = oldprogname;
}


static int handle_script (lua_State *L, char **argv, int n) {
  int status;
  const char *fname;
  int narg = getargs(L, argv, n);  /* collect arguments */
  lua_setglobal(L, "arg");
  fname = argv[n];
  if (strcmp(fname, "-") == 0 && strcmp(argv[n-1], "--") != 0)
    fname = NULL;  /* stdin */
  status = luaL_loadfile(L, fname);
  lua_insert(L, -(narg+1));
  if (status == LUA_OK)
    status = docall(L, narg, LUA_MULTRET);
  else
    lua_pop(L, narg);
  return report(L, status);
}


/* check that argument has no extra characters at the end */
#define noextrachars(x)		{if ((x)[2] != '\0') return -1;}


/* indices of various argument indicators in array args */
#define has_i		0	/* -i */
#define has_v		1	/* -v */
#define has_e		2	/* -e */
#define has_E		3	/* -E */

#define num_has		4	/* number of 'has_*' */


static int collectargs (char **argv, int *args) {
  int i;
  for (i = 1; argv[i] != NULL; i++) {
    if (argv[i][0] != '-')  /* not an option? */
        return i;
    switch (argv[i][1]) {  /* option */
      case '-':
        noextrachars(argv[i]);
        return (argv[i+1] != NULL ? i+1 : 0);
      case '\0':
        return i;
      case 'E':
        args[has_E] = 1;
        break;
      case 'i':
        noextrachars(argv[i]);
        args[has_i] = 1;  /* go through */
      case 'v':
        noextrachars(argv[i]);
        args[has_v] = 1;
        break;
      case 'e':
        args[has_e] = 1;  /* go through */
      case 'l':  /* both options need an argument */
        if (argv[i][2] == '\0') {  /* no concatenated argument? */
          i++;  /* try next 'argv' */
          if (argv[i] == NULL || argv[i][0] == '-')
            return -(i - 1);  /* no next argument or it is another option */
        }
        break;
      default:  /* invalid option; return its index... */
        return -i;  /* ...as a negative value */
    }
  }
  return 0;
}


static int runargs (lua_State *L, char **argv, int n) {
  int i;
  for (i = 1; i < n; i++) {
    lua_assert(argv[i][0] == '-');
    switch (argv[i][1]) {  /* option */
      case 'e': {
        const char *chunk = argv[i] + 2;
        if (*chunk == '\0') chunk = argv[++i];
        lua_assert(chunk != NULL);
        if (dostring(L, chunk, "=(command line)") != LUA_OK)
          return 0;
        break;
      }
      case 'l': {
        const char *filename = argv[i] + 2;
        if (*filename == '\0') filename = argv[++i];
        lua_assert(filename != NULL);
        if (dolibrary(L, filename) != LUA_OK)
          return 0;  /* stop if file fails */
        break;
      }
      default: break;
    }
  }
  return 1;
}


static int handle_luainit (lua_State *L) {
  const char *name = "=" LUA_INITVERSION;
  const char *init = getenv(name + 1);
  if (init == NULL) {
    name = "=" LUA_INIT;
    init = getenv(name + 1);  /* try alternative name */
  }
  if (init == NULL) return LUA_OK;
  else if (init[0] == '@')
    return dofile(L, init+1);
  else
    return dostring(L, init, name);
}


static int pmain (lua_State *L) {
  int argc = (int)lua_tointeger(L, 1);
  char **argv = (char **)lua_touserdata(L, 2);
  int script;
  int args[num_has];
  args[has_i] = args[has_v] = args[has_e] = args[has_E] = 0;
  if (argv[0] && argv[0][0]) progname = argv[0];
  script = collectargs(argv, args);
  if (script < 0) {  /* invalid arg? */
    print_usage(argv[-script]);
    return 0;
  }
  if (args[has_v]) print_version();
  if (args[has_E]) {  /* option '-E'? */
    lua_pushboolean(L, 1);  /* signal for libraries to ignore env. vars. */
    lua_setfield(L, LUA_REGISTRYINDEX, "LUA_NOENV");
  }
  /* open standard libraries */
  luaL_checkversion(L);
  lua_gc(L, LUA_GCSTOP, 0);  /* stop collector during initialization */
  luaL_openlibs(L);  /* open libraries */
#ifdef lua_userinit
  lua_userinit(L);   /* APRIL: init packages */
#endif
  lua_gc(L, LUA_GCRESTART, 0);
  if (!args[has_E] && handle_luainit(L) != LUA_OK)
    return 0;  /* error running LUA_INIT */
  /* execute arguments -e and -l */
  if (!runargs(L, argv, (script > 0) ? script : argc)) return 0;
  /* execute main script (if there is one) */
  if (script && handle_script(L, argv, script) != LUA_OK) return 0;
  if (args[has_i])  /* -i option? */
    dotty(L);
  else if (script == 0 && !args[has_e] && !args[has_v]) {  /* no arguments? */
    if (lua_stdin_is_tty()) {
      print_version();
      dotty(L);
    }
    else dofile(L, NULL);  /* executes stdin as a file */
  }
  lua_pushboolean(L, 1);  /* signal no errors */
  return 1;
}


int main (int argc, char **argv) {
  int status, result;
  lua_State *L = luaL_newstate();  /* create state */
  if (L == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  /* call 'pmain' in protected mode */
  lua_pushcfunction(L, &pmain);
  lua_pushinteger(L, argc);  /* 1st argument */
  lua_pushlightuserdata(L, argv); /* 2nd argument */
  status = lua_pcall(L, 2, 1, 0);
  result = lua_toboolean(L, -1);  /* get result */
  finalreport(L, status);
  lua_close(L);
  return (result && status == LUA_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}

