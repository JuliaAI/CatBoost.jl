using PyCall

let dependencies = ["catboost==0.24.8"]
    pip = pyimport("pip")
    pip.main(["install"; split(get(ENV, "PIPFLAGS", "")); dependencies])
end
