import sys
import os
from modules.yolo_tracker import run_yolo_tracker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CYAN  = "\033[96m"
YELLOW= "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _banner():
    print(f"""
{CYAN}{BOLD}
                                   *                                      
                                   *                                       
                             **        **                                  
                          *                 *                              
                        *                     *                            
                       *                       *                           
                                   *                                       
                  ****           *****            ****                      
                                   *                                        
                       *                       *                           
                         *                   *                             
                           **             **                               
                                   *                                        
                                   *                                       
                                                                           
                                                                           
                                                               
{RESET}""")


def _ask_camera():
    raw = input(f"  {YELLOW}Indice webcam{RESET} [default=0]: ").strip()
    return int(raw) if raw.isdigit() else 0


def main():
    _banner()
    cam = _ask_camera()
    run_yolo_tracker(camera_index=cam)


if __name__ == "__main__":
    main()