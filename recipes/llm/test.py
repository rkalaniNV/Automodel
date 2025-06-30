# document_paths = [f"/lustre/fsw/coreai_dlalgo_nemofw/dpykhtar/dclm/preprocessed/dclm_{i+1:02d}_text_document"for i in range(50)]
# index_mapping_dir = "/lustre/fsw/coreai_dlalgo_llm/aot/tmp/rp2_index_mapping/"
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset

prefix = "/lustre/fsw/coreai_dlalgo_nemofw/dpykhtar/dclm/preprocessed/dclm_01_text_document"  # <- drop .bin/.idx
ds = make_dataset(prefix, impl="mmap")      # or impl="cached" / "lazy"
print(len(ds))          # number of samples
print(ds[0])            # numpy array of token-ids in the first sample
breakpoint()