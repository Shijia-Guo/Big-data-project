package data_preparation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

public class generate_trainning_file {

	public static void main(String[] args) throws IOException {
		if(args.length < 3){
			System.out.println("The arguments should be: InputFolder outputFile type subType(is optional)");
			return;
		}
		String inputFolder = args[0];
		String outputFile = args[1];
		String type = args[2];
		String subType = "";
		if(args.length == 4)
			subType = args[3];
		List<String> records = getAllTxtFilePath(inputFolder);
		File fout = new File(outputFile);
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for(String item : records){
        	bw.write(extract_single_train_record(item,type,subType));
        	
        }
        bw.close();
	}
	public static List<String> getAllTxtFilePath(String directoryName){
	    File f = null;  
	    f = new File(directoryName);  
	    File[] files = f.listFiles();  
	    List<String> list = new ArrayList<String>();  
	    for (File file : files) {  
	        if(file.isDirectory()) {  
	        	list.addAll(getAllTxtFilePath(file.getAbsolutePath()));  
	        } else {
	        	if(file.getAbsolutePath().contains("HumanMethylation") && file.getAbsolutePath().endsWith("txt"))
	        		list.add(file.getAbsolutePath());  
	        }  
	    }  
	    return list;
	}
	public static String extract_single_train_record(String fileName, String type, String subType) {
        File file = new File(fileName);
        BufferedReader reader = null;
        StringBuffer sb = new StringBuffer();
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            boolean  header = true;
           
            while ((tempString = reader.readLine()) != null) {
            	if(header){
            		header = false;
            		continue;
            	}
            	String[] result = tempString.split("\\s+");
            	if("NA".equalsIgnoreCase(result[1]))
            		sb.append(-1);
            	else
            		sb.append(result[1]);
            	sb.append(" ");
            }
            sb.append(type);
            if(!subType.isEmpty())
            	sb.append(" ").append(subType);
            sb.append("\n");
            reader.close();
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
        return sb.toString();
    }	
}
