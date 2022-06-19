for i in {1..5}; do
    echo -e "\033[34mbuild/bin/tiny test/tiny/parser/test_${i}.tiny -emit=ast\033[0m";
    build/bin/tiny test/tiny/parser/test_${i}.tiny -emit=ast;
done

echo -e "\033[31mbuild/bin/tiny test/tiny/parser/test_6.tiny -emit=jit\033[0m";
build/bin/tiny test/tiny/parser/test_6.tiny -emit=jit;

echo -e "\033[33mbuild/bin/tiny test/tiny/parser/test_7.tiny -emit=mlir -opt\033[0m";
build/bin/tiny test/tiny/parser/test_7.tiny --emit=mlir -opt;

echo -e "\033[34mbuild/bin/tiny test/tiny/parser/test_7.tiny -emit=ast\033[0m";
build/bin/tiny test/tiny/parser/test_7.tiny -emit=ast;

echo -e "\033[31mbuild/bin/tiny test/tiny/parser/test_7.tiny -emit=jit\033[0m";
build/bin/tiny test/tiny/parser/test_7.tiny -emit=jit;