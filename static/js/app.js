(function() {
	var name;

	guess_country = function(url, callback) {
		var req;
		req = $.getJSON(url);
		return req.success(function(data) {
			return callback(data)
		});
	};

	disp = function(data) {
		$("#results h3").remove()
		var results
		results = $("#results")
		for (_i = 0, _len = data.length; _i < _len; _i++) {
			results.append('<h3>' + data[_i].country + '</h3>')
		}
	}

	send_request = function() {
		name = $("#name").val()
		return guess_country("/guess/" + name, disp);
	}

	$("#guess").click(send_request)

	$('#name').keypress(function (e) {
  		if (e.which == 13) {
	    	$('#guess').click();
	    	return false;
		}
	});

	send_request()

}).call(this)
