Inputs:
	$Content: ChannelT[$$Channels, TensorT[$$Dimensions]]
	$Style: ChannelT[$$Channels, TensorT[$$Dimensions]]

Output: ChannelT[$$Channels, TensorT[$$Dimensions]]

Parameters:
	$Epsilon: Defaulting[ScalarT, 10^-5]


AllowDynamicDimensions: True


SowMeanSigma[input_, eps_] := Scope[
	mean = SowNode["mean", input, "axis" -> {2, 3}, "keepdims" -> True];
	var = SowSquare@SowBMinus[input, mean];
	var = SowNode["mean", var, "axis" -> {2, 3}, "keepdims" -> True];
	sigma = SowSqrt@SowBPlus[var, eps];
	Return@{mean, sigma}
]


Writer: Function[
	ctx = GetInput["Content", "Batchwise"];
	sty = GetInput["Style", "Batchwise"];
	{sMean, sSigma} = SowMeanSigma[sty, #Epsilon];
	{cMean, cSigma} = SowMeanSigma[ctx, #Epsilon];
	sub = SowBMinus[ctx, cMean];
	div = SowBDivide[sSigma, cSigma];
	out = SowBPlus[SowBHad[sub, div], sMean];
	SetOutput["Output", out];
]

Suffix: "Layer"