if [ $# -ne 2 ]
then
	print "Usage of the script : ./compare_files.sh file1(precision mayor) file2"
	print "File1 debe contener los numeros de mayor precision, para luego"
	print "compararse con los datos de file usando un pequenyo epsilon"
	exit
fi
cat $1 | head -n -2 | tail -n +5 > comp.txt
cat $2 | head -n -2 | tail -n +5 | LANG=C awk -v file1=comp.txt '
function abs(x){return (((x < 0.0) ? -x : x) + 0.0)}
BEGIN{
	i=1;
	num_line=1;
	num_errors=0;
  epsilon=0.01;
	while(getline <file1)
	{
		for(j=1;j<=NF;j++)
		{
			line[i","j]=$j;
		}
		i++;
	}
}
{
	for(i=1;i<=NF;i++)
	{
		true_value=abs(line[num_line","i]);
		max_value=true_value+true_value*epsilon;
		min_value=true_value-true_value*epsilon;
		if (abs($i) > max_value || abs($i) < min_value)
		{
			printf("Error at comparing numbers precision: %f %f\n", true_value, $i);
      printf("%f %f\n", min_value, max_value);
			num_errors++;
		}
	}
	num_line++;
}
END{printf("Numero de errores: %d\n", num_errors);
if(num_errors == 0)
	printf("Todo esta OK.\n")
else
	printf("Parece que algun numero falla...\n")
}'
rm comp.txt
