PREFIX = /usr
UNAME := `uname`
LINUX_SUFIX := $(shell ( ldconfig -p 2>/dev/null | grep libmkl_core > /dev/null && echo "mkl" ) || ( ls /opt/MKL/lib/libmkl_core.so 2> /dev/null > /dev/null && echo "mkl" ) || ( ldconfig -p 2>/dev/null | grep libatlas > /dev/null && echo "atlas" ) || echo "")

ifeq ($(UNAME),"Darwin")
# homebrew
ifneq ("$(wildcard /usr/local/include/lua.h)","")
DARWIN_SUFIX = homebrew
PREFIX = /usr/local
endif
# macports
ifneq ("$(wildcard /opt/local/include/lua.h)","")
DARWIN_SUFIX = macports
PREFIX = /opt/local
endif
endif

INCLUDE := $(PREFIX)/include
LIB := $(PREFIX)/lib
LUALIB := $(PREFIX)/lib/lua/5.2
LUAMOD := $(PREFIX)/share/lua/5.2
BIN := $(PREFIX)/bin

ALL: auto-release

########################## AUTOMACTIC SECTION ############################

auto-release:
	@echo "System $(UNAME)"
	@make system-release-$(UNAME)

auto-debug:
	@echo "System $(UNAME)"
	@make system-debug-$(UNAME)

auto-test-debug:
	@echo "System $(UNAME)"
	@make system-test-debug-$(UNAME)

# RELEASE

system-release-Linux: check_linux_release
	@echo "Sufix $(LINUX_SUFIX)"
	@make release-$(LINUX_SUFIX)

system-release-Darwin: check_darwin_release
	@echo "Sufix $(DARWIN_SUFIX)"
	@make release-$(DARWIN_SUFIX)

# DEBUG

system-debug-Linux: check_linux_release
	@echo "Sufix $(LINUX_SUFIX)"
	@make debug-$(LINUX_SUFIX)

system-debug-Darwin: check_darwin_release
	@echo "Sufix $(DARWIN_SUFIX)"
	@make debug-$(DARWIN_SUFIX)

# TEST-DEBUG

system-test-debug-Linux: check_linux_release
	@echo "Sufix $(LINUX_SUFIX)"
	@make test-debug-$(LINUX_SUFIX)

system-test-debug-Darwin: check_darwin_release
	@echo "Sufix $(DARWIN_SUFIX)"
	@make test-debug-$(DARWIN_SUFIX)

# CHECK

check_linux_release:
ifeq ("$(LINUX_SUFIX)", "")
	@echo "Impossible to detect the proper release!"
	@exit 1
endif

check_darwin_release:
ifeq ("$(DARWIN_SUFIX)", "")
	@echo "Impossible to detect macports or homebrew!"
	@exit 2
endif

#############################################################################

release: auto-release

debug: auto-debug

test: auto-test-debug

document:
	rm -Rf doxygen_doc build_doc
	lua profile_build_scripts/build_release_atlas.lua document

performance:
	april-ann TEST/PERFORMANCE/register_performance.lua TEST/PERFORMANCE/matrix/test.lua

#############################################################################

# TEST with OMP and ATLAS
test-debug-atlas: debug-atlas
	lua profile_build_scripts/build_debug_atlas.lua test

# TEST without OMP and ATLAS
test-debug-no-omp: debug-no-omp
	lua profile_build_scripts/build_debug_no_omp.lua test

# TEST for DARWIN MACPORTS
test-debug-macports: debug-macports
	lua profile_build_scripts/build_debug_macports.lua test

# TEST for DARWIN HOMEBREW
test-debug-homebrew: debug-homebrew
	lua profile_build_scripts/build_debug_homebrew.lua test

# TEST with OMP and MKL
test-debug-mkl: debug-mkl
	lua profile_build_scripts/build_debug_mkl.lua test

# TEST with CUDA and MKL
test-debug-cuda-mkl: debug-cuda-mkl
	lua profile_build_scripts/build_debug_cuda_and_mkl.lua test

# TEST for raspberry-pi (subset of packages)
test-debug-pi: debug-pi
	lua profile_build_scripts/build_debug_pi.lua test

#############################################################################

# RELEASE for DARWIN MACPORTS
release-macports:
	lua profile_build_scripts/build_release_macports.lua

# RELEASE for DARWIN HOMEBREW
release-homebrew:
	lua profile_build_scripts/build_release_homebrew.lua

# RELEASE with OMP and MKL
release-mkl:
	lua profile_build_scripts/build_release_mkl.lua

# RELEASE with OMP and ATLAS
release-atlas:
	lua profile_build_scripts/build_release_atlas.lua

# RELEASE with CUDA and MKL
release-cuda-mkl:
	lua profile_build_scripts/build_release_cuda_and_mkl.lua

# RELEASE for raspberry-pi (subset of pacakges)
release-pi:
	lua profile_build_scripts/build_release_pi.lua

# RELEASE without OMP and ATLAS
release-no-omp:
	lua profile_build_scripts/build_release_no_omp.lua

#############################################################################

# DEBUG for DARWIN MACPORTS
debug-macports:
	lua profile_build_scripts/build_debug_macports.lua

# DEBUG for DARWIN HOMEBREW
debug-homebrew:
	lua profile_build_scripts/build_debug_homebrew.lua

# DEBUG with OMP and MKL
debug-mkl:
	lua profile_build_scripts/build_debug_mkl.lua

# DEBUG with OMP and ATLAS
debug-atlas:
	lua profile_build_scripts/build_debug_atlas.lua

# DEBUG without OMP and ATLAS
debug-no-omp:
	lua profile_build_scripts/build_debug_no_omp.lua

# DEBUG with CUDA and MKL
debug-cuda-mkl:
	lua profile_build_scripts/build_debug_cuda_and_mkl.lua

# DEBUG for raspberry-pi (subset of packages)
debug-pi:
	lua profile_build_scripts/build_debug_pi.lua

#############################################################################

INCLUDE_NO_WS := $(shell echo "$(INCLUDE)" | sed 's/ //g')
LIB_NO_WS := $(shell echo "$(LIB)" | sed 's/ //g')
LUALIB_NO_WS := $(shell echo "$(LUALIB)" | sed 's/ //g')
LUAMOD_NO_WS := $(shell echo "$(LUAMOD)" | sed 's/ //g')
BIN_NO_WS := $(shell echo "$(BIN)" | sed 's/ //g')

check-env-vars:
ifneq ("$(INCLUDE_NO_WS)","$(INCLUDE)")
	@echo "Unable to work with whitespaces in PREFIX, INCLUDE, LIB, LUALIB, LUAMOD, BIN env vars"
	@exit 3
endif
ifneq ("$(LIB_NO_WS)","$(LIB)")
	@echo "Unable to work with whitespaces in PREFIX, INCLUDE, LIB, LUALIB, LUAMOD, BIN env vars"
	@exit 4
endif
ifneq ("$(LUALIB_NO_WS)","$(LUALIB)")
	@echo "Unable to work with whitespaces in PREFIX, INCLUDE, LIB, LUALIB, LUAMOD, BIN env vars"
	@exit 5
endif
ifneq ("$(LUAMOD_NO_WS)","$(LUAMOD)")
	@echo "Unable to work with whitespaces in PREFIX, INCLUDE, LIB, LUALIB, LUAMOD, BIN env vars"
	@exit 6
endif
ifneq ("$(BIN_NO_WS)","$(BIN)")
	@echo "Unable to work with whitespaces in PREFIX, INCLUDE, LIB, LUALIB, LUAMOD, BIN env vars"
	@exit 7
endif

clean:
	./clean.sh

install: check-env-vars uninstall
	@mkdir -p $(LUAMOD)
	@mkdir -p $(LUALIB)
	rsync -r tools/ $(LUAMOD)/april_tools
	@mkdir -p $(INCLUDE)/april-ann
	install -m 444 include/april-ann/* $(INCLUDE)/april-ann
	install -m 444 include/april-ann.h $(INCLUDE)/april-ann.h
	@sed "s#__PREFIX__#$(PREFIX)#g" .april-ann.pc > april-ann.pc
	install april-ann.pc $(LIB)/pkgconfig/april-ann.pc
	@rm april-ann.pc
	install lib/libapril-ann.a $(LIB)
	install lib/libapril-ann.so $(LIB)
	install lib/aprilann.so $(LUALIB)
	install bin/april-ann $(BIN)

uninstall: check-env-vars
	@rm -Rf $(INCLUDE)/april-ann
	@rm -Rf $(LUAMOD)/april_tools
	@rm -f $(INCLUDE)/april-ann.h
	@rm -f $(LUALIB)/aprilann.so
	@rm -f $(LIB)/libapril-ann.a
	@rm -f $(LIB)/libapril-ann.so
	@rm -f $(LIB)/pkgconfig/april-ann.pc
	@rm -f $(BIN)/april-ann

##############################################################################

.PHONY: all
