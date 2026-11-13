import os
import glob
import time
import subprocess

def reset_ansys_environment():
    print("\033[94mIn function reset_ansys_environment.\033[0m")
    print("\033[94m{\033[0m")

    
    subprocess.run(
        ["taskkill", "/F", "/IM", "ansysedt.exe", "/T"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    subprocess.run(
        ["taskkill", "/F", "/IM", "AnsysGRPC.exe", "/T"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    
    ansoft_dir = r"C:\Users\Surface\Documents\Ansoft"
    for f in glob.glob(os.path.join(ansoft_dir, "*.aedt.auto")):
        try: 
            os.remove(f)
        except Exception: 
            pass
            
    time.sleep(1)

    print("\033[94mIn function reset_ansys_environment: Successfully killed Ansys tasks.\033[0m")
    print("\033[94m}\033[0m")
    print("\033[94m\033[0m")

if __name__ == "__main__":
    reset_ansys_environment()