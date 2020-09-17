import os
import pickle


class Debugger(object):
    def __init__(self, dir: str=".temp", reset: bool= False):
        assert os.path.exists(dir), "{} doesn't exist".format(dir)
        self.dir = dir
        if reset:
            self.reset()

    def save(self, ob, name: str=None):
        save(self.dir, ob, name)

    def load(self, name: str=None):
        return load(self.dir, name)

    def reset(self):
        reset(self.dir)


def save(dir: str, ob, name: str=None):
    if not name:
        with open(os.path.join(dir, "debug.info"), "r") as file:
            num = int(file.readline().strip())
        name = str(num+1) + ".pkl"
        with open(os.path.join(dir, "debug.info"), "w") as file:
            file.write(str(num+1))
    with open(os.path.join(dir, name), "wb") as file:
        pickle.dump(ob, file)


def load(dir: str, name: str=None):
    if not name:
        with open(os.path.join(dir, "debug.info"), "r") as file:
            num = int(file.readline().strip())
        name = str(num) + ".pkl"
    else:
        assert os.path.exists(os.path.join(dir, name)), "{} doesn't exist".format(name)

    with open(os.path.join(dir, name), "rb") as file:
        ob = pickle.load(file)
    return ob


def reset(dir: str):
    for name in os.listdir(dir):
        os.remove(os.path.join(dir, name))
    with open(os.path.join(dir, "debug.info"), "w") as file:
        file.write("0")


def info(data):
    return "sp: {} ty: {} dv: {} max: {} min: {}".format(data.shape, data.dtype, data.device, data.max(), data.min())


if __name__ == '__main__':

    debugger = Debugger(reset=True)
    a = 1234567
    debugger.save(a, "rpn_locs.pkl")
    b = debugger.load("rpn_locs.pkl")

    print("End.")