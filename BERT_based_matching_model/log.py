import math

def main():
    y = 1
    p = 0.3917
    print(f'p: {p}')
    if p > 0:
        log_p = math.log(p, 2)
        print(f'log_p: {log_p}')
    else:
        log_p = 0
        
    if 1-p > 0:
        log_1_p = math.log(1-p, 2)
        print(f'log_1_p: {log_1_p}')
    else:
        log_1_p = 0
        
    BCE_loss = -(y * log_p + (1-y) * log_1_p)
    print(f'BCE_loss: {BCE_loss}')

if __name__ == "__main__":
    main()