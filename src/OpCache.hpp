template <typename T, typename O>
struct Op {
    static T op (T x) {
        return O::opImpl(x);
    }
};

struct ExpOp : Op<float, ExpOp> {
    static float opImpl (float x) {
        return std::exp(x);
    }
};

template <typename T, typename O>
struct OpCache {
    T last_in, last_out;
    OpCache(T init) : last_in(init), last_out(O::op(init)) {}
    T get(T in) {
        if (in != last_in) {
            last_in = in;
            last_out = O::op(in);
            std::cout << "new: ";
        } else {
            std::cout << "cached: ";
        }
        return last_out;
    }
};

template <typename P>
struct Param {
    Param() {}
    float get() {
        return static_cast<P*>(this)->getImpl();
    };
};


template <typename T, typename P, typename O>
struct ParamCache {
    P param;
    OpCache<T, O> opCache;
    ParamCache() : param(), opCache(param.get()) {}
    T get() {
        return opCache.get(param.get());
    }
};


/* EXAMPLE USAGE:
template <size_t index>
struct VectorParam : Param<VectorParam<index>> {
    VectorParam() {}
    float getImpl() {
        return params[index];
    };
};

template <size_t indexParam, size_t indexInput>
struct CVInput : Param<CVInput<indexParam, indexInput>> {
    CVInput() {}
    float getImpl() {
        return params[indexParam] + inputs[indexInput];
    };
};

[...]

    ParamCache<float, CVInput<FREQUENCY_PARAM, FREQUENCY_INPUT>, ExpOp> expCache;
    expCache.get();

*/