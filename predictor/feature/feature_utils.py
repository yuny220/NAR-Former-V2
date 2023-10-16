# define the op and its attributes to be encoded
OPS = {
    "Conv": {
        "code": 1,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
            "dilations",
            "group",
            "bias",
        ],
    },
    "Relu": {
        "code": 2,
        "attrs": [],
    },
    "Add": {
        "code": 3,
        "attrs": [],
    },
    "Sigmoid": {
        "code": 4,
        "attrs": [],
    },
    "Reshape": {
        "code": 5,
        "attrs": [],
    },
    "MaxPool": {
        "code": 6,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
            #"dilations",
        ],
    },
    "Split": {
        "code": 7,
        "attrs": [],
    },
    "GlobalAveragePool": {
        "code": 8,
        "attrs": [],
    },
    "Gemm": {
        "code": 9,
        "attrs": [
            "bias",
        ],
    },
    "Transpose": {
        "code": 10,
        "attrs": [
            "perm",
        ]
    },
    "Upsample": {
        "code": 11,
        "attrs": [],
    },
    "BatchNormalization": {
        "code": 12,
        "attrs": [],
    },
    "Mul": {
        "code": 13,
        "attrs": [],
    },
    "Concat": {
        "code": 14,
        "attrs": [],
    },
    "Flatten": {
        "code": 15,
        "attrs": [],
    },
    "AveragePool": {
        "code": 16,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
        ],
    },
    "Cast": {
        "code": 17,
        "attrs": [],
    },
    "Matmul": {
        "code": 18,
        "attrs": [],
    },
    "ReduceMean": {
        "code": 19,
        "attrs": [],
    },
    "Pow": {
        "code": 20,
        "attrs": [],
    },
    "Slice": {
        "code": 21,
        "attrs": [],
    },
    "Div": {
        "code": 22,
        "attrs": [],
    },
    "Sub": {
        "code": 23,
        "attrs": [],
    },
    "Sqrt": {
        "code": 24,
        "attrs": [],
    },
    "Clip": {
        "code": 25,
        "attrs": [],
    },
    "Softmax": {
        "code": 26,
        "attrs": [],
    },
    "Tanh": {
        "code": 27,
        "attrs": [],
    },
    "ConvTranspose": {
        "code": 28,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
            "dilations",
            "group",
            "bias",
            "output_padding",
        ],
    },
}

# define the value type of attr value, and its feature length
ATTRS = {
    "kernel_shape"  : ("tuple", 1,  0,   1),
    "strides"       : ("tuple", 1,  0,   1),
    "pads"          : ("tuple", 1,  0,   1),
    "dilations"     : ("tuple", 1,  0,   1),
    "group"         : ("int"  ,     0,   1),
    "bias"          : ("bool" ,     0,   1),
    "perm"          : ("tuple", 8,  0,   1),
    "output_padding": ("tuple", 1,  0,   1),
}

# define the fixed length of feature op_code, attrs, output shape
FEATURE_LENGTH = {                                                                                                                                           
    "op_type": 32,                                                                                                                                            
    "attrs": 8,                                                                                                                                              
    "output_shape": 4,                                                                                                                                       
    "topology": 2,                                                                                                                                           
    "static": 4,                                                                                                                                             
}        

# dim= 152 = 32 + 10*8 + 10*4
FEATURE_DIM = {                                                                                                                                                                                                                                                                                    
    "attrs": 80,                                                                                                                                            
    "output_shape": 40,                                                                                                                                                                                                                                                                            
    "static": 40,                                                                                                                                           
}