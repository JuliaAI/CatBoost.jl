module DatasetsAPI

using CatBoost
using DataFrames

train, test = load_dataset(:msrank_10k)

end # module
