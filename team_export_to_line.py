from subprocess import run, PIPE, STDOUT

for i in range(0, 287):
	with open('construct_teams/team_folder/team_export_'+str(i)+'.json', 'r') as file:
		team_data_i = file.read().replace('\n', '')

	bashCommand = "./pokemon-showdown pack-team"
	process = run(bashCommand.split(), input = team_data_i, encoding='ascii', stdout=PIPE)
	if(process.returncode == 0):
		file_out = open("construct_teams/team_folder/team_" + str(i) + '.txt',"w")
		file_out.write(process.stdout) 
		file_out.close() 
	else:
		ValueError("Invalid team given")