# Variables definition 
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

# Automatic dependencies management
-include make.dep
make.dep: $(SRCS)
	@$(RM) make.dep
	@for f in $(SRCS); do \
		$(CXX) $(CPPFLAGS) -MM $$f >> make.dep; \
	done

# Compilation rule from .o to executable
main: $(OBJS)
	 @$(CXX) -o $@ $^

# Clean function defintion
.PHONY: clean
clean:
	@rm -f $(OBJS) make.dep


all: make.dep main clean
