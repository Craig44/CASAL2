#' Utility extract function
#'
#' @author Craig Marsh
#' @description 
#' create a matrix that has a header
#' @keywords internal
#'
"make.matrix_with_header" <-
function(lines)
{
  columns <- string.to.vector.of.words(lines[1])
  if(length(lines) < 2) 
    return(NA)
  data <- matrix(0, length(lines) - 1, length(columns))
  for(i in 1:length(lines)) {
      line = string.to.vector.of.numbers(lines[i])
      if (length(line) != length(columns)) {
          stop(paste(lines[i],"is not the same length as",lines[1]))
      }
    data[i - 1,  ] <- line
  }
  #colnames(data) <- columns
  data
}