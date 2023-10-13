#cd-hit_seq_similarity_filter

###usage

输入一个带有多个单序列fasta文件的文件夹，输出文件夹为序列相似度（cluster）小于某个值的所有的fasta文件


###parameter
      data_path: { type: str, help: 'fasta files dir path'}

      output_dir: { type: str, help: 'output dir', default : "output_dir"}
      
      cluster: { type: float, help: "序列相似度", default : 0.4} 

      memory: { type: int, help: 'memory usage', default :  9600}

      threads: { type: int, help: 'number of threads', default : 8}



###example

手动 python make_fasta.py生成文件夹fasta_files，里面是利用mini_soluprot.csv内的每条蛋白序列单独生成fasta文件

mlflow run seq_similarity_filter.csv -P data_path="/path/to/fasta_files/"

输出一个含有所有相似度低于某值的fasta文件的文件夹


