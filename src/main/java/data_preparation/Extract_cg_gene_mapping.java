package data_preparation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class Extract_cg_gene_mapping {

	public static void main(String[] args) {
		generate_cfg_gene_mapping("C:\\Users\\anyone\\workspace\\Pre-process\\SARC.txt");

	}
	
	public static void generate_cfg_gene_mapping(String fileName) {
        File file = new File(fileName);
        BufferedReader reader = null;
        Map<String, Set<String>> mappings  = new HashMap<>();
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            int line = 0;
            while ((tempString = reader.readLine()) != null) {
            	// skip the column header
            	line++;
            	if(line == 1)
            		continue;
            	String[] result = tempString.split("\\s+");
            	if(result[5].length() <=1)
            		continue;
            	
            	String[] genes = result[5].split(";");
            	Set<String> unique = new HashSet<>();
            	for(int j = 0 ; j < genes.length;j++){
            		unique.add(genes[j]);
            	}
            	mappings.put(result[0],unique);
                
            }
            reader.close();
            File fout = new File("cfg_gene_mappings.txt");
            FileOutputStream fos = new FileOutputStream(fout);
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
            for(Map.Entry<String, Set<String>> entry: mappings.entrySet())
            {
            	String key = entry.getKey();
            	String value = " ";
            	Iterator<String> it = entry.getValue().iterator();  
            	while (it.hasNext()) {  
            	  String str = it.next();  
            	  value = value + str + ";";
            	}  
            	value = value.substring(0, value.length()-1);
            	bw.write(key+value+"\n");
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
    }
}
