import tarfile
from pydub import AudioSegment


def extract(files_to_extract):
    count = 0
    with tarfile.open(r"C:\Users\hadis\Downloads\en.tar") as archive:
        for member in archive:
            last = member.name.split("/")
            if last[len(last) - 1] in files_to_extract:
                count += 1
                print(f"Found file. Count: {count} / {len(files_to_extract)}")
                archive.extract(member, path=r"I:\Gender")
                print(f"Finished {last}")

    print("done")


def convert_mp3_to_wav(file: str):
    AudioSegment.from_mp3(file).export()
