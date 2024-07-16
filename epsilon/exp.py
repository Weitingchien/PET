import math







def main():

    s_p = [0.0195, 0.0250, 0.0079, 0.0811, -0.0462]

    sum_exp = 0
    for x in s_p:
        sum_exp += math.exp(x)

    print(sum_exp)
    print(f'e^0.0195: {math.exp(s_p[0])}')
    print(1.0196/5.0923)


if __name__ == "__main__":
    main()