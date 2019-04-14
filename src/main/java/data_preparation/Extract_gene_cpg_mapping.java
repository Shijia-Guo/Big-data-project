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

public class Extract_gene_cpg_mapping {

	public static void main(String[] args) {
		generate_gene_cpg_mapping("C:\\Users\\anyone\\workspace\\cs5425_course\\cpg_gene_mappings.txt");
	}
	
	public static void generate_gene_cpg_mapping(String fileName) {
        File file = new File(fileName);
        BufferedReader reader = null;
        Map<String, Set<String>> mappings  = new HashMap<>();
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {
            	String[] result = tempString.split("\\s+");
            	String[] genes = result[1].split(";");
            	
            	for(int i = 0 ; i < genes.length ;i++){
            		if(mappings.containsKey(genes[i])){
            			mappings.get(genes[i]).add(result[0]);
            		}else{
            			Set<String> unique = new HashSet<>();
            			unique.add(result[0]);
            			mappings.put(genes[i], unique);
            		}
            	}
            }
            reader.close();
            File fout = new File("gene_cpg_mappings.txt");
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
