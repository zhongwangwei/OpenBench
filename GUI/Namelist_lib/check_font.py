import os
import shutil
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.font_manager as fm
import subprocess

class check_font():
    def __init__(self):
        self.author = "Qingchen Xu/xuqingchen23@163.com"
        self.font_file = matplotlib.matplotlib_fname()
        self.font_path = os.path.join(os.path.dirname(self.font_file),'fonts/ttf')
        self.font_list = [font_manager.FontProperties(fname=font).get_name().lower() for font in font_manager.findSystemFonts(fontpaths=self.font_path, fontext='ttf')]
        self.font_families = sorted(set([f.name.lower() for f in fm.fontManager.ttflist]))
        self.find_path = './GUI/Namelist_lib/Font/'
        self.find_list = [font_manager.FontProperties(fname=font).get_name().lower() for font in
                          font_manager.findSystemFonts(fontpaths=self.find_path, fontext='ttf')]
        self.find_family = [font for font in font_manager.findSystemFonts(fontpaths=self.find_path, fontext='ttf')]

    def check_font(self, font):
        if font.lower() not in self.font_list:
            st.warning(f"Font '{font}' not in  matplotlib，Trying to find font from {self.find_path}...")
            if font.lower() in self.find_list:
                st.warning('find font in path')
                index = self.find_list.index(font.lower())
                file = self.find_family[index]

                try:
                    shutil.copy(file, self.font_path)
                    subprocess.run('fc -cache',shell=True)
                    subprocess.run('rm ~/.cache/matplotlib/ -rf',shell=True)
                    st.warning(f"'{font}' available!。")
                except Exception as e:
                    st.warning(f"Can not find font：{font}, please change to others!")
            else:
                st.warning(f"Can not find font：{font}, please change to others!")

