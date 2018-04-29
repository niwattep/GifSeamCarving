import CAIS
import time

def function_test1():
    CAIS.CAIS('in.gif', (200, 200), 'function.gif', True, True)

def function_test2():
    CAIS.CAIS_forloop('in.gif', (200, 200), 'function_forloop.gif', True, True)

def test1():
    CAIS.verbose = False
    t0 = time.time()
    print("test 1 processing...")
    CAIS.CAIS('in.gif', (150, 200), 'test1_out.gif', False, True)
    print("time: ", time.time() - t0)

def test1_forloop():
    CAIS.verbose = False
    t0 = time.time()
    print("test 1 forloop processing...")
    CAIS.CAIS('in.gif', (150, 200), 'test1_forloop_out.gif', False, True)
    print("time: ", time.time() - t0)

def test2():
    CAIS.verbose = False
    t0 = time.time()
    print("test 2 processing...")
    CAIS.CAIS('in.gif', (150, 150), 'test2_out.gif', False, True)
    print("time: ", time.time() - t0)

def test2_forloop():
    CAIS.verbose = False
    t0 = time.time()
    print("test 2 forloop processing...")
    CAIS.CAIS('in.gif', (150, 150), 'test2_forloop_out.gif', False, True)
    print("time: ", time.time() - t0)

if __name__ == "__main__":
    function_test1() #just a test to ensure that function can run properly
    function_test2() #just a test to ensure that function can run properly

    # time test with multiprocessing and for loop
    test1()
    test1_forloop()
    test2()
    test2_forloop()