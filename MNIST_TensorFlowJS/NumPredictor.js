function NumPredictorApp ()
{
    // Drawing canvas stuff...

    // get canvas and 2D context and set him correct size
    var canvas = document.getElementById ('drawableArea');
    var ctx = canvas.getContext('2d');

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect (0, 0, canvas.width, canvas.height);

    // last known position
    var pos = { x: 0, y: 0 };

    document.addEventListener ('mousemove', draw);
    document.addEventListener ('mousedown', setPosition);
    document.addEventListener ('mouseenter', setPosition);
    document.getElementById ('clearBtn').addEventListener ('click', clear);
    document.getElementById ('predictBtn').addEventListener ('click', predict);


    // D3 graph stuff...

    // SVG Width, height, and some added spacing
    var margin = {
            top:    15,
            right:  5,
            bottom: 10,
            left:   5
    };

    var graphWidth  = 395 - margin.left - margin.right;
    var graphHeight = 200 - margin.top - margin.bottom;
    var barLabels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ];

    // Empty, for now
    var dataset, xScale, yScale;
    var updateDuration = 500; 

    // Define the scales to convert our data to screen coordinates
    xScale = d3.scaleLinear ()
               .domain ( [ 0, 10 ] )
               .range ( [ 0, graphWidth ] ); 

    yScale = d3.scaleLinear ()
               .domain ( [ 0, 100 ] )
               .range ( [ graphHeight, 10 ] );

    var svg = d3.select ('#predGraph')
                .append ('svg')
                .attr ('width', graphWidth + margin.left + margin.right)   
                .attr ('height', graphHeight + 2*margin.top + 2*margin.bottom);

    xAxis = d3.axisBottom ()
              .scale (xScale)
              .ticks (10)
              .tickSizeInner ( 10 )
              .tickSizeOuter ( 10 )
              .tickPadding ( -1 )
              .tickFormat ( function (d, i) { return barLabels[i]; });

    // Define Y axis
    yAxis = d3.axisLeft ()
              .scale (yScale)
              .ticks (10);

    // Create axes..
    svg.append ('g')
       .attr ('class', 'axis')
       .attr ('transform', 'translate(0,' + (graphHeight) + ')')
       .call (xAxis)
       .selectAll ('text')  
       .style ('text-anchor', 'middle')
       .attr ('dx', xScale(0.5) );

    svg.append ('text')
       .style ('text-anchor', 'middle')
       .text ('Prediction Precentages')
       .attr ('x', xScale (5))
       .attr ('y', graphHeight +  2.5*margin.bottom + 10)
       .attr ('class', 'legend');
 
    var space    = xScale(1.03) - xScale(1);
    var barWidth = xScale(1.95) - xScale(1);

    var dataset = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];

    // Make the bars...
    for (var i = 0; i < 10; i++)
    {
        var xValue = 1.25*space + xScale(i);
        var yValue = yScale (dataset[i]);

        svg.append ('rect')
           .attr ('id',     ('rect_' + i) )
           .attr ('x',      xValue )
           .attr ('y',      yValue )
           .attr ('width',  barWidth )
           .attr ('height', yScale(0) - yValue )
           .attr ('fill',   '#333333' ) 
           .attr ('tooltipText', dataset[i].toFixed(3) + ' %')
           .on ('mouseover', function (d)
           {
               var xPosition = parseFloat (d3.select(this).attr('x')) + (xScale(1) - xScale(0)) / 2;
               var yPosition = parseFloat (d3.select(this).attr('y')) + Math.abs(yScale(0) - yScale(d)) / 2;

               xPosition = d3.event.pageX; //d3.mouse(this)[0];
               yPosition = d3.event.pageY; //d3.mouse(this)[1];

               d3.select ('#tooltip')
                 .style ('left', xPosition + 'px')
                 .style ('top', (yPosition - 15) + 'px')
                 .select ('#label').html ( d3.select (this).attr ('tooltipText') );

               d3.select ('#tooltip').classed ('hidden', false);
           })
           .on ('mouseout', function (d)
           {
               d3.select ('#tooltip').classed ('hidden', true);
           }) 
    }
    
    //
    // UpdateGraph: utility function to update the graph.  Note that the dataset needs 
    //              to updated first
    // 
    function UpdateGraph ()
    {
        for (var i = 0; i < 10; i++)
        { 
            var yValue = yScale (dataset[i]);
                   
            d3.select ('#rect_' + i) 
              .transition ()
              .duration (updateDuration)
              .attr ('y',      yValue )
              .attr ('height', (yScale(0) - yValue) )
              .attr ('tooltipText', dataset[i].toFixed(3) + ' %' );
        }
    }

    //
    // setPosition: Used with the number drawing 
    //
    function setPosition (event)
    {
        // Get the bounding rectangle of target
        const rect = document.getElementById('contents').getBoundingClientRect();

        pos.x = event.clientX - rect.left;//event.pageX;//clientX;
        pos.y = event.clientY - rect.top;//event.pageY;//clientY;
    }

    //
    // clear: callback for the clear button
    //
    function clear ()
    {
        // clear the drawing area
        ctx.clearRect (0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect (0, 0, canvas.width, canvas.height);

        // reset the graph to all zeros...
        dataset = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
        UpdateGraph ();

        // set the prediction number to nothing...
        document.getElementById ('predictedNum').innerText = '';
    }

    //
    // predict: callback for the predict button...
    //
    async function predict ()
    {
        // Load the pretrained model...
        const model = await tf.loadLayersModel ('Model_CNN/model.json');

        // uncomment for testing/debugging
        // model.summary ();


        // Prepare the data...
    
        // Get the drawn number...
        let imageData = ctx.getImageData (0, 0, canvas.width, canvas.height);
        let img = tf.browser.fromPixels (imageData); 
 
        // prep the number image so that the CNN can understand what it should do 
        // with it...

        // resize the image it so its 28 x 28
        img = tf.image.resizeBilinear (img, [28, 28]).toFloat();

        // adjust the dimensions so that the image will be accepted by the CNN... note
        // that the CNN requires tensors in the form [ batch number, image x, image y, channel ];
        // so the tensor needs to be resized to [ 1, 28, 28, 1 ]
        img  = img.mean (2)
                  .toFloat ()
                  .expandDims (0)
                  .expandDims (-1);

        // Finally normalize the data... 
        img = img.div (tf.scalar (255.0));

        // and take the 'inverse' of the image since the CNN was trained with white numbers on a
        // black background and we have black numbers on a white background
        img = tf.scalar (1.0).sub (img);


        // Make the prediction...
        results = model.predict(img).arraySync()[0]; 

        // uncomment for testing/debugging
        // console.log ('Predictions: ', model.predict(img).print());

        // Using ES6's '...' operator to implement the argMax function
        const indexOfMaxValue = results.indexOf (Math.max (...results)); 

        // Update the predicted number text...
        document.getElementById ('predictedNum').innerText = indexOfMaxValue;

        // make the values percentages... so multiply by 100
        for (var i = 0; i < 10; i++)
            dataset[i] = results[i] * 100;
               
        // update the graph
        UpdateGraph ();
    }

    //
    // draw: callback to handle drawing in the canvas...
    //
    function draw(e)
    {
        // mouse left button must be pressed
        if (e.buttons !== 1) 
            return;

        // begin
        ctx.beginPath(); 

        ctx.lineWidth = 30;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#000';

        ctx.moveTo (pos.x, pos.y); // from
        setPosition (e);
        ctx.lineTo (pos.x, pos.y); // to

        // draw it!
        ctx.stroke(); 
    }

}



