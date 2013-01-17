
cmdOptTest = cmdOpt{
  program_name = "ls",
  argument_description = "file or path",
  main_description = "List  information  about  the  FILEs  (the  current directory by default).",
  author = "Written by Donald duck.",
  copyright = "GPL license",
  see_also = "info ls",
  -- options can be added in the constructor:
  { index_name="author",
    description = "print the author of each file",
    long = "author",
    short = "l",
    argument = "no",
  },
  { index_name="blocks-size",
    description = "use SIZE-byte blocks",
    long = "blocks-size",
    argument = "yes",
    argument_name = "size",
  },
  { index_name="entries_by_columns",
    description = "list entries by columns",
    short = "C",
    argument = "no",
  },
  { index_name="save-one",
    description = "estoy testeando",
    long = "save-one",
    argument = "no",
  },
  { index_name="save-two",
    description = "estoy testeando 2",
    long = "save-two",
    mode="always",
    argument = "optional",
  },
  { index_name="directory",
    description = "list directory entries instead of contents, and do not dereference symbolic links",
    short = "d",
    long = "directory",
    argument = "no",
  },
}
-- options can also be declared outside constructor:
cmdOptTest:add_option{
  description = "shows this help message",
  short = "h",
  long = "help",
  argument = "no",
  action = function (argument) 
	     print(cmdOptTest:generate_help()) 
	     os.exit(1)
	   end
}
cmdOptTest:add_option{
  index_name = "ignore-backups",
  description = "do not list implied entries ending with ~",
  short = "B",
  long = "ignore-backups",
  argument = "no",
  action = function(value) print("this is a test "..tostring(value)) end,
}
cmdOptTest:add_option{
  index_name = "ignore",
  description = "do not list implied entries matching shell PATTERN",
  short = "I",
  long = "ignore",
  argument = "yes",
  argument_name = "PATTERN",
}

----------------------------------------------------------------------
---------------------------- MAIN PROGRAM ----------------------------
----------------------------------------------------------------------
result = cmdOptTest:parse_args()
if type(result) == 'string' then error(result) end

print("\nThis is a test, let's see the contents of result table:")
for i,j in pairs(result) do
  print(i,j)
end

