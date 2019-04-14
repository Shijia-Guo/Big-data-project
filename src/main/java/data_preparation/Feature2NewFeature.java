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
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Feature2NewFeature {

	public static void main(String[] args) throws Exception {
		if(args.length<6)
			throw new Exception("Application requires: features.txt, cpgname.txt, cpg_gene_mappings.txt, gene_cpg_mappings,full train data file, output train file");
		
		List<Integer> indexes = extract_feature_index(args[0]);
		List<String> cpgnames = indexes_to_cpg_name(indexes,args[1]);
		List<String> new_cpgnames = cpg_transform(cpgnames,args[2],args[3]);
		generate_new_trainning_data(args[4],args[5],args[1],new_cpgnames);
	}
	public static List<Integer> extract_feature_index(String old_feature_file) throws IOException{
		 	File file = new File(old_feature_file);
	        BufferedReader reader = null;
	        List<Integer> indexes = new LinkedList<>();
	        reader = new BufferedReader(new FileReader(file));
	        String tempString = null;
	        while ((tempString = reader.readLine()) != null) {
	           String[] result = tempString.split("\\s+");
	           indexes.add(Integer.parseInt(result[1])); 	
	         }
	        reader.close();
	        return indexes;

	}
	public static List<String> indexes_to_cpg_name(List<Integer> indexes, String cpgname_file) throws IOException{
		File file = new File(cpgname_file);
        BufferedReader reader = null;
        List<String> cpgnames = new LinkedList<>();
        Map<Integer, String> index_name = new HashMap<>();
        reader = new BufferedReader(new FileReader(file));
        String tempString = null;
        int index = 0;
        while ((tempString = reader.readLine()) != null) {
        	index_name.put(index, tempString);
        	index++;
         }
        reader.close();
        for(Integer item : indexes){
        	cpgnames.add(index_name.get(item));
        }
        return cpgnames;
		
	}
	public static List<String> cpg_transform(List<String> cpgnames,String cpg_gene_mappings_file,String gene_cpg_mappings_file) throws IOException{
		Map<String, String> cpg_gene_mapping = new HashMap<>();
        Map<String, String> gene_cpg_mapping = new HashMap<>();
        Set<String> genes = new HashSet<>();
        Set<String> enrich_cpg_names = new HashSet<>();
		File file = new File(cpg_gene_mappings_file);
        BufferedReader reader = null;
        reader = new BufferedReader(new FileReader(file));
        String tempString = null;
        while ((tempString = reader.readLine()) != null) {
        	String[] res = tempString.split("\\s+");
        	cpg_gene_mapping.put(res[0],res[1]);
         }
        reader.close();
        
        file = new File(gene_cpg_mappings_file);
        reader = null;
        reader = new BufferedReader(new FileReader(file));
        
        while ((tempString = reader.readLine()) != null) {
        	String[] res = tempString.split("\\s+");
        	gene_cpg_mapping.put(res[0],res[1]);
         }
        reader.close();
        
        for(String cpg : cpgnames){
        	if(cpg_gene_mapping.containsKey(cpg)){
        		String single_cpg_genes[] = cpg_gene_mapping.get(cpg).split(";"); 
        		for(int i = 0 ;i < single_cpg_genes.length;i++){
        			genes.add(single_cpg_genes[i]);
        		}
        	}else{
        		enrich_cpg_names.add(cpg);
        	}
        }
        // persist the genes list
        File fout = new File("genes.txt");
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for(String gene: genes){
        	bw.write(gene+"\n");
        }
        bw.close();
        for(String gene: genes){
        	if(gene_cpg_mapping.containsKey(gene)){
        		String single_gene_cpgs[] = gene_cpg_mapping.get(gene).split(";"); 
        		for(int i = 0 ;i < single_gene_cpgs.length;i++){
        			enrich_cpg_names.add(single_gene_cpgs[i]);
        		}
        	}
        }
        //persist new cpgnames
        fout = new File("enrich_cpg.txt");
        fos = new FileOutputStream(fout);
        bw = new BufferedWriter(new OutputStreamWriter(fos));
        for(String cpg: enrich_cpg_names){
        	bw.write(cpg+"\n");
        }
        bw.close();
        return new LinkedList<>(enrich_cpg_names);
	}
	
	public static void generate_new_trainning_data(String full_train_file,String train_file, String cpgname_file,List<String> cpgnames) throws IOException{
		File file = new File(cpgname_file);
        BufferedReader reader = null;
        Map<String,Integer> name_index = new HashMap<>();
        reader = new BufferedReader(new FileReader(file));
        String tempString = null;
        int index = 0;
        while ((tempString = reader.readLine()) != null) {
        	name_index.put(tempString,index);
        	index++;
         }
        reader.close();
        
        List<Integer> indexes = new LinkedList<>();
        for(String name: cpgnames){
        	indexes.add(name_index.get(name));
        }
        File new_train_file = new File(train_file);
        FileOutputStream fos = new FileOutputStream(new_train_file);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        StringBuffer s = new StringBuffer();
        for(String name: cpgnames){
        	s.append(name).append(" ");
        }
        s.append("label\n");
        bw.write(s.toString());
        File fullTrainfile = new File(full_train_file);
        reader = new BufferedReader(new FileReader(fullTrainfile));
        while ((tempString = reader.readLine()) != null) {
        	s.delete(0,s.length());
        	String res[] = tempString.split("\\s+");
        	for(int i = 0 ; i < indexes.size(); i++){
        		s.append(res[indexes.get(i)]).append(" ");
        	}
        	s.append(res[res.length-1]+"\n");
        	bw.write(s.toString());
         }
        bw.close();
        reader.close();
	}
}
