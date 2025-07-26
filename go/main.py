from rl.simulate import experience_simulation
from agent.naive_fast import FastRandomBot


def main():
    res = experience_simulation(10, FastRandomBot(), FastRandomBot())
    print(res)


if __name__ == '__main__':
    main()