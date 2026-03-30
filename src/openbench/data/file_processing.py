import os
import shutil
import logging
class FileProcessing:
   def __init__(self):
      self.name = 'DatasetPreprocessing'
      self.version = '0.1'
      self.release = '0.1'
      self.date = 'Mar 2023'
      self.author = "Zhongwang Wei / zhongwang007@gmail.com"
      # copy all the dict in info to self
   
   def check_file_exist(self, file_path):
      if not os.path.exists(file_path):
         logging.error(f"File not found: {file_path}")
         raise FileNotFoundError(f"File not found: {file_path}")
      return 
   
   def check_dir_exist(self, dir_path):
      if not os.path.exists(dir_path):
         logging.error(f"Directory not found: {dir_path}")
         raise FileNotFoundError(f"Directory not found: {dir_path}")
      return 
   
   def create_dir(self, dir_path):
      if not os.path.exists(dir_path):
         os.makedirs(dir_path)
      return 
   
   def remove_file(self, file_path):
      if os.path.exists(file_path):
         os.remove(file_path)
      return 
   
   def remove_dir(self, dir_path):
      if os.path.exists(dir_path):
         shutil.rmtree(f'{dir_path}', ignore_errors=True)
      return 
   

   def mk_dir(self,basedir,evaluation_only,comparison_only,comparisons,statistics,scores,metrics):
      self.basedir = basedir

      if evaluation_only or comparison_only:
         pass
      else:      
         self.remove_dir (f'{self.basedir}')
         self.create_dir (f'{self.basedir}')
         self.scratch_dir = os.path.join(f'{self.basedir}', 'scratch')
         self.create_dir (f'{self.scratch_dir}')
         self.tmp_dir = self.scratch_dir  # Use scratch for all temporary files

         self.output_dir     = f'{self.basedir}'
         self.outputdata_dir = os.path.join(f'{self.output_dir}','data')
         self.create_dir (f'{self.outputdata_dir}')

         # if self.score_vars is not empty, then create the score directory
         if scores:
            self.score_dir = os.path.join(f'{self.output_dir}', 'scores')
            self.create_dir (f'{self.score_dir}')
         if metrics:
            self.metric_dir = os.path.join(f'{self.output_dir}',  'metrics')
            self.create_dir (f'{self.metric_dir}')

         for st in statistics:
            self.statistics_dir = os.path.join(f'{self.output_dir}', 'statistics',f'{st}')
            self.create_dir (f'{self.statistics_dir}')

         for comparison in comparisons:
            self.comparison_dir = os.path.join(f'{self.output_dir}', 'comparisons', f'{comparison}')
            self.create_dir (f'{self.comparison_dir}')

      return

   def mk_list(self,file_path,list_patch,dsource):
      filenames = sorted(glob.glob(f'{file_path}*.nc'))
      df = pd.DataFrame(columns=['ID', 'SYEAR', 'EYEAR', 'LON', 'LAT', 'file_path'], index=range(len(filenames)))  # 'R': R,'KGE': KGE,
      for i, file in enumerate(filenames):
         with xr.open_dataset(f'{file}') as ds:
            df.iloc[i, 0] = file.split('_')[0]
            df.iloc[i, 1] = ds.time.dt.year[0].values
            df.iloc[i, 2] = ds.time.dt.year[-1].values
            df.iloc[i, 3] = ds.lon.values
            df.iloc[i, 4] = ds.lat.values
            df.iloc[i, 5] = path
      df.to_csv(f"{list_patch}/{dsource}_tmp.csv", index=False)
