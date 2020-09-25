import configparser
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys

def xml_to_csv(xml_path):
    xml_list = []
    for xml_file in glob.glob(xml_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    data_path = sys.argv[1]
    for folder in ['1_train_data', '2_eval_data']:
        xml_path = os.path.join(data_path, folder, 'xml')
        csv_output_path = os.path.join(data_path, folder, 'tfrecord')
        if not os.path.isdir(csv_output_path):
            os.makedirs(csv_output_path)
        
        xml_df = xml_to_csv(xml_path)
        xml_df.to_csv(csv_output_path + '/' + folder + '.csv', index=None)
        print('Successfully converted ' + folder + 'xml to csv.')

