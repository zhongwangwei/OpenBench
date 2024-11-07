source ~/anaconda3/etc/profile.d/conda.sh
conda activate streamlit
conda info --envs

echo "If your connected by 'ssh -L 8000:127.0.0.1:8501 tms15', then copy http://127.0.0.1:8000/"
echo "Otherwise please copy the 'Network URL' to your browser."
streamlit run ./GUI/GUI_openbench.py