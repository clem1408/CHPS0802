if(EXISTS "/home/clem/CHPS0802/TP1/build/tests[1]_tests.cmake")
  include("/home/clem/CHPS0802/TP1/build/tests[1]_tests.cmake")
else()
  add_test(tests_NOT_BUILT tests_NOT_BUILT)
endif()
