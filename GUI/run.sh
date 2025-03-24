#!/bin/bash
home_directory=$HOME
default_env_file="$home_directory/.default_env"

# 读取用户输入
read_input() {
    read -p "$1 [yes/y or no/n]: " response
    case $response in
        [Yy][Ee][Ss]|[Yy])
            return 0
            ;;
        [Nn][Oo]|[Nn])
            return 1
            ;;
        *)
            echo "Invalid input, please enteryes、y、no or n。"
            read_input "$1"
            ;;
    esac
}

Start_streamlit(){
SERVER_ADDRESS="127.0.0.1"
SERVER_PORT="8000"

# 提示用户输入地址和端口
echo "If you not connect by ssh -L, Press Enter to Pass. "
read -p "Please input Streamlit server address [Default: 127.0.0.1]: " input_address
read -p "Please input Streamlit server port [Default: 8000]: " input_port

# 如果用户输入了值，则覆盖默认值
if [ -n "$input_address" ]; then
    SERVER_ADDRESS="$input_address"
fi

if [ -n "$input_port" ]; then
    SERVER_PORT="$input_port"
fi

# 运行 Streamlit
echo "Start Streamlit server, Address: $SERVER_ADDRESS, Port: $SERVER_PORT"
echo "If you Connect by ssh -L, copy Connect URL http://$SERVER_ADDRESS:$SERVER_PORT/ to your browser"
echo "Other wise copy Network URL to your browser"
streamlit run ./GUI/GUI_openbench.py
}

# 设置默认环境
set_default_env() {
    read -p "Please enter the name of the Anaconda environment that you want to set as the default environment: " env_name
    if [ -z "$env_name" ]; then
        echo "The environment name cannot be empty"
        exit 1
    else
        echo "$env_name" > "$default_env_file"
        echo "The default environment is set: $env_name"
    fi
}


select_env() {
  # 提示用户是否使用自己安装的 Anaconda 环境
read_input "Whether or not to use your own installed Anaconda environment"
use_anaconda=$?

if [ $use_anaconda -eq 0 ]; then
    read -p "Please enter the name of the Anaconda environment: " env_name
    if [ -z "$env_name" ]; then
        echo "The environment name cannot be empty"
        exit 1
    else
        if conda info --envs | grep -q "$env_name"; then
          echo "The name of the Anaconda environment used is: $env_name"
          source "$home_directory/anaconda3/etc/profile.d/conda.sh"
          conda activate "$env_name"
          read_input "Whether to set this environment as the default environment"
          if [ $? -eq 0 ]; then
              set_default_env
          fi
          Start_streamlit
        else
            echo "Environment '$env_name' does not exist."
            read_input "Please enter input environment again '$env_name'?"
            if [ $? -eq 0 ]; then
                echo "The name of the Anaconda environment used is: $env_name"
                source "$home_directory/anaconda3/etc/profile.d/conda.sh"
                conda activate "$env_name"
                conda info --envs
                read_input "Whether to set this environment as the default environment"
                if [ $? -eq 0 ]; then
                    set_default_env
                fi
                Start_streamlit
            else
                echo "Environment creation skipped. Exiting."
                exit 1
            fi
        fi
    fi
else
    file_path="$HOME/.bashrc_streamlit"
    if [ -f "$file_path" ]; then
        source "$home_directory/.bashrc_streamlit"
    else
        echo "Don't find .bashrc_streamlit file。"
        cp "/home/xuqch3/.bashrc_streamlit" "$HOME/.bashrc_streamlit"
    fi
    source "$home_directory/.bashrc_streamlit"
    conda activate Openbench
    conda info --envs
    Start_streamlit
fi
}

# 检查是否设置了默认环境
check_default_env() {
    if [ -f "$default_env_file" ]; then
        default_env=$(cat "$default_env_file")
        echo "The default environment was detected: $default_env"
        read_input "Whether to use the default environment"
        if [ $? -eq 0 ]; then
            source "$home_directory/anaconda3/etc/profile.d/conda.sh"
            conda activate "$default_env"
            echo "Default environment activated: $default_env"
            conda info --envs
            Start_streamlit
        else
            echo "Continue to select a different environment."
            select_env
        fi
    else
        echo "The default environment is not set."
        select_env
    fi
}

check_default_env





















##echo "If your connected by 'ssh -L 8000:127.0.0.1:8501 tms15', then copy http://127.0.0.1:8000/"
##echo "Otherwise please copy the 'Network URL' to your browser."
##streamlit run streamlit_app.py





##source ~/anaconda3/etc/profile.d/conda.sh
##conda activate streamlit
##conda info --envs
##
##echo "If your connected by 'ssh -L 8000:127.0.0.1:8501 tms15', then copy http://127.0.0.1:8000/"
##echo "Otherwise please copy the 'Network URL' to your browser."
##streamlit run ./GUI/GUI_openbench.py