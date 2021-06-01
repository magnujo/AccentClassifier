import os
import tarfile
from pydub import AudioSegment


def extract(files_to_extract, save_dir: str, tar_location: str):
    count = 0
    with tarfile.open(tar_location) as archive:
        for member in archive:
            last = member.name.split("/")
            if last[len(last) - 1] in files_to_extract:
                count += 1
                print(f"Found file. Count: {count} / {len(files_to_extract)}")
                archive.extract(member, path=save_dir)
                print(f"Finished {last}")

    print("done")


def convert_mp3s_to_wavs(path_to_mp3s: str):
    os.chdir(path_to_mp3s)

    audio_files = os.listdir()
    count = 0
    n = len(audio_files)
    # You dont need the number of files in the folder, just iterate over them directly using:
    for file in audio_files:
        # spliting the file into the name and the extension
        name, ext = os.path.splitext(file)
        if ext == ".mp3":
            mp3_sound = AudioSegment.from_mp3(file)
            # rename them using the old name + ".wav"
            print(f"Converting {file} to wav. Count: {count} / {n}")
            mp3_sound.export("{0}.wav".format(name), format="wav")
            os.remove(file)
        count += 1
    print("done")