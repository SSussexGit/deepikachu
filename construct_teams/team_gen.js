const Koffing = require('koffing').Koffing;
var fs = require('fs');
var textByLine = fs.readFileSync('gen5ou.txt').toString();

const teamCode = textByLine;

let parsedTeam = Koffing.parse(teamCode);
console.log(JSON.stringify(parsedTeam['teams'][0]['pokemon']));

function sum( obj ) {
  var sum = 0;
  for( var el in obj ) {
    if( obj.hasOwnProperty( el ) ) {
      sum += parseFloat( obj[el] );
    }
  }
  return sum;
}
//checking if a team has sensible evs
// for (i = 0; i < parsedTeam['teams'].length; i++){
// 	console.log("-------");
// 	console.log(i);
// 	console.log("-----");
// 	for (j = 0; j < 6; j++){
// 		//console.log(parsedTeam['teams'][i]['pokemon'][j]["evs"]);
// 		// if (sum(parsedTeam['teams'][i]['pokemon'][j]["evs"]) < 500){
// 		// 	console.log("alert");
// 		// 	console.log(j);
// 		// }
// 		//console.log(sum(parsedTeam['teams'][i]['pokemon'][j]["evs"]));
// 	}
// }

for (i = 0; i < parsedTeam['teams'].length; i++){
	fs.writeFile("team_folder/team_export_"+ i + ".json", JSON.stringify(parsedTeam['teams'][i]['pokemon']), function(err) {
	    if (err) {
	        console.log(err);
	    }
	});
}
