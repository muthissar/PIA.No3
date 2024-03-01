import glob
import re
import shutil
import string
import subprocess
import os
import io


def convert_database(database_source, database_dest):
    # Get files list
    midi_files = glob.glob(database_source + '/**/*.mid')
    for midi_file in midi_files:
        # Create write folder
        source_dir = os.path.dirname(midi_file)
        dest_dir = re.sub(database_source, database_dest, source_dir)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        #  Get write path
        write_path_mxml = re.sub(r'\.mid$', '.xml', midi_file)
        write_path_mxml = re.sub(database_source, database_dest, write_path_mxml)
        cmd = "mscore " + '\"' + midi_file + '\"' + " -o " + '\"' + write_path_mxml + '\"'
        subprocess.call(cmd, shell=True)


def filter_str(s):
    printable = set(string.printable)
    list_filter = filter(lambda x: x in printable, s)
    ret = ''.join(list_filter)
    return ret


def change_instrument_names(source_folder, target_folder):
    """
    Change the part-name field of musicxml files by splitting it in two part,
    assigning one to part-name field and the other to instrument-name field.
    Useful for xml files generated using musescore
    :param source_folder:
    :param target_folder:
    :return:
    """
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    list_folders = glob.glob(source_folder + '/**')

    for folder in list_folders:
        # List mxml
        xml_files = glob.glob(folder + '/*.xml')
        for xml_file in xml_files:
            try:
                # Read file
                list_line = []
                with io.open(xml_file, 'r', encoding="utf-8") as ff:
                    for line in ff:
                        line_filtered = filter_str(line)
                        match_part = re.search(r"<part-name>(.*)</part-name>", line_filtered)
                        if match_part:
                            part_name = match_part.group(1)
                            try:
                                new_instru_name, new_part_name = re.split(', ', part_name)
                            except:
                                new_part_name = part_name
                                new_instru_name = part_name
                            line_filtered = re.sub(part_name, new_part_name, line_filtered)

                        match_instru = re.search(r"<instrument-name>(.*)</instrument-name>", line_filtered)
                        if match_instru:
                            instrument_name = match_instru.group(1)
                            line_filtered = re.sub(instrument_name, new_instru_name, line_filtered)
                        list_line.append(line_filtered)

                # Write back files
                target_file = re.sub(source_folder, target_folder, xml_file)
                folder_out = re.sub(source_folder, target_folder, os.path.dirname(xml_file))
                if not os.path.isdir(folder_out):
                    os.makedirs(folder_out)

                with open(target_file, 'w') as ff:
                    for line in list_line:
                        ff.write(line)
            except:
                print("########################")
                print(xml_file)


if __name__ == '__main__':
    # name = 'imslp'
    # convert_database(database_source=f'/home/leo/Recherche/Databases/Orchestration/arrangement_midi/{name}',
    #                  database_dest=f'/home/leo/Recherche/Databases/Orchestration/arrangement_mxml/{name}_temp')
    # change_instrument_names(source_folder=f'/home/leo/Recherche/Databases/Orchestration/arrangement_mxml/{name}_temp',
    #                         target_folder=f'/home/leo/Recherche/Databases/Orchestration/arrangement_mxml/{name}')


    # convert_database(database_source=f'/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works_clean',
    #                  database_dest=f'/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works_clean_mxml')
    change_instrument_names(source_folder=f'/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works_clean_mxml_temp',
                            target_folder=f'/home/leo/Recherche/Databases/Orchestration/BACKUP/Kunstderfuge/Selected_works_clean_mxml')
