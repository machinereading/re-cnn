package edu.kaist.mrlab.nn.pcnn.rest;

import javax.ws.rs.Consumes;
import javax.ws.rs.FormParam;
import javax.ws.rs.OPTIONS;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Response;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

@Path("/re-pcnn")
public class REPOSTService {

	@SuppressWarnings("unchecked")
	@POST
	@Consumes("application/json; charset=UTF-8")
	@Produces("application/json; charset=UTF-8")
	// @Produces("text/plain; charset=UTF-8")
	public Response getPost(String input) throws Exception {

		// @FormParam("text")

//		JSONParser jsonParser = new JSONParser();
//		JSONObject reader = (JSONObject) jsonParser.parse(input);
//		String text = (String) reader.get("text");

		// String result = edu.kaist.mrlab.nn.pcnn.rest.Main.t.singleRun(text);
		edu.kaist.mrlab.nn.pcnn.rest.Main.t.run(true, "data/test/ko_test_parsed.txt", "");

		// System.out.println(result);
		//
		// JSONObject resultObj = new JSONObject();
		// resultObj.put("result", result);
		// resultObj.toString()

		return Response.ok().entity("").header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				.header("Access-Control-Allow-Origin", "*")
				.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept").build();

	}

	@OPTIONS
	@Consumes("application/x-www-form-urlencoded; charset=utf-8")
	@Produces("text/plain; charset=utf-8")
	public Response getOptions(@FormParam("text") String input) throws Exception {

		return Response.ok().header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				.header("Access-Control-Allow-Origin", "*")
				.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept").build();

	}

}
