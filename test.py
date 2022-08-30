def abs(x):
    if x>0:
        return x
    return -x

def sqrtd(a):
    if a < 0:
        print('error')
        return 
    else:
        x = 1
        alpha = 0.001
        deta = 1
        count = 1
        while abs(deta) > 1e-10:
            deta = 4*x*(x**2-a)

            x -= alpha*deta
            count += 1

    return x

if __name__ == '__main__':
    print(sqrtd(25))