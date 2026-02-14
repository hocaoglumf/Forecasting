import tkinter as tk
from tkinter import filedialog

def pick_file(title="Bir dosya seçin", filetypes=None):
    root = tk.Tk()
    root.withdraw()          # ana pencereyi gizle
    root.attributes("-topmost", True)  # pencere önde açılsın (opsiyonel)

    path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes or [("Tüm dosyalar", "*.*")]
    )
    root.destroy()
    return path

if __name__ == "__main__":
    file_path = pick_file(
        title="Veri dosyasını seç",
        filetypes=[("CSV dosyaları", "*.csv"), ("Excel", "*.xlsx *.xls"), ("Tüm dosyalar", "*.*")]
    )
    print("Seçilen dosya:", file_path if file_path else "(iptal edildi)")
